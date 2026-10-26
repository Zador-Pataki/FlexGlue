"""Export FlexGlue to ONNX for COLMAP and check it against PyTorch.

Two graphs are exported:

- flexglue_descriptor.onnx
    image [1, 3, S, S] float32 in [0, 1] -> features [1, 256, S/8, S/8]
  The image is resized to S x S (squashed, bilinear, no antialiasing) by the
  caller, as DINOv3Descriptor does with a (h, w) resize.

- flexglue_matcher.onnx (sparse mode, batch size 1)
    features0, features1          [1, 256, S/8, S/8] float32
    kpts0, kpts1                  [1, N0, 2], [1, N1, 2] float32, pixels with
                                  the origin at the top-left image corner
    image_size0, image_size1      [1, 2] float32, (w, h) of the original image
    filter_threshold              scalar float32
    max_distance_px               scalar float32, inf disables the check
    min_covisibility              scalar float32, 0 disables the check
  ->
    matches0                      [1, N0] int64, index into kpts1 or -1
    mscores0                      [1, N0] float32, 0 where unmatched
    covisibility_score            [1] float32
  All matches are rejected (matches0 = -1) if
  covisibility_score < min_covisibility.

The matcher runs all transformer layers (no early stopping). The thresholds
are graph inputs, so that they can be tuned at runtime without re-exporting.

By default, the attention of the transformer layers is exported as the fused
com.microsoft::MultiHeadAttention contrib operator of ONNX Runtime (flash /
memory-efficient attention on CUDA) instead of being decomposed into
MatMul/Softmax/MatMul, which materializes the attention matrices. Such models
require ONNX Runtime; use --no_fused_attention for standard ONNX operators.

Two further export-time rewrites, mathematically identical to the reference
implementation, avoid slow paths of ONNX Runtime:
- The rows of the fused qkv projection are permuted from (H, D, 3) to
  (3, H, D) and the attention is computed in the token-major (B, N, H, D)
  layout, which avoids splitting q, k, v along an innermost axis of size 3
  and the transposes around the fused attention.
- The argmax of the correspondence readout is taken from the (already
  computed) top-k instead of ArgMax/ReduceMax, which run on slow cuDNN
  reductions.

Requires onnx, onnxscript and onnxruntime (onnxruntime-gpu for --check on GPU).

Usage (from the root of the repository):
    python onnx/export.py --check            # descriptor + fp32 matcher
    python onnx/export.py --check --fp16 --matcher_only  # fp16 matcher
The models are written to onnx/models/ by default. See onnx/match_pair.py for
an example of inference with ONNX Runtime.
"""

import argparse
import contextlib
import copy
import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxscript
import torch
import torch.nn.functional as F
from onnx.defs import OpSchema
from torch import nn

# Also run from a clone of the repository without installing it.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from flexglue import DINOv3Descriptor, FlexGlue  # noqa: E402
from flexglue.flexglue import WEIGHTS_URL  # noqa: E402
from flexglue.utils import load_image, modules, ops  # noqa: E402
from flexglue.utils import preprocess as pp  # noqa: E402
from flexglue.utils.modules import PairContext  # noqa: E402

# Minimal schema of ONNX Runtime's contrib MultiHeadAttention operator (only
# the query/key/value inputs and the output), so that the exporter can emit it.
onnx.defs.register_schema(
    OpSchema(
        "MultiHeadAttention",
        "com.microsoft",
        1,
        inputs=[
            OpSchema.FormalParameter("query", "T"),
            OpSchema.FormalParameter("key", "T"),
            OpSchema.FormalParameter("value", "T"),
        ],
        outputs=[OpSchema.FormalParameter("output", "T")],
        type_constraints=[("T", ["tensor(float)", "tensor(float16)"], "")],
        attributes=[OpSchema.Attribute("num_heads", OpSchema.AttrType.INT, "")],
    )
)
MSFT_OPSET = onnxscript.values.Opset("com.microsoft", 1)


@torch.library.custom_op("flexglue::multi_head_attention", mutates_args=())
def multi_head_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Attention on (B, N, H*D) tensors, as com.microsoft::MultiHeadAttention."""
    b, n, e = query.shape
    d = e // num_heads
    q, k, v = (
        t.view(b, t.shape[1], num_heads, d).transpose(1, 2) for t in (query, key, value)
    )
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(b, n, e)


@multi_head_attention.register_fake
def _(query, key, value, num_heads):
    return torch.empty_like(query)


def multi_head_attention_onnx(query, key, value, num_heads: int):
    return MSFT_OPSET.MultiHeadAttention(query, key, value, num_heads=num_heads)


def fused_scaled_dot_product_attention(q, k, v):
    """Drop-in for F.scaled_dot_product_attention on (B, H, N, D) tensors."""
    b, h, n, d = q.shape

    def to_bne(t):
        return t.transpose(1, 2).reshape(b, t.shape[2], h * d)

    out = torch.ops.flexglue.multi_head_attention(to_bne(q), to_bne(k), to_bne(v), h)
    return out.view(b, n, h, d).transpose(1, 2)


class _FusedAttentionFunctional:
    """torch.nn.functional with a fused scaled_dot_product_attention."""

    def __getattr__(self, name):
        if name == "scaled_dot_product_attention":
            return fused_scaled_dot_product_attention
        return getattr(F, name)


@contextlib.contextmanager
def fused_attention(enabled: bool = True):
    """Route the attention of the FlexGlue transformer through the custom op."""
    original = modules.F
    if enabled:
        modules.F = _FusedAttentionFunctional()
    try:
        yield
    finally:
        modules.F = original


def _attention_bnhd(q, k, v, fused: bool):
    """Attention on (B, N, H, D) tensors, returns (B, N, H*D)."""
    if fused:
        h = q.shape[-2]
        return torch.ops.flexglue.multi_head_attention(
            q.flatten(-2), k.flatten(-2), v.flatten(-2), h
        )
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2).flatten(-2)


def _self_block_forward(self, x, encoding, fused):
    """SelfBlock.forward with token-major (B, N, H, D) layout.

    Expects the rows of Wqkv permuted to the (3, H, D) order (see
    permute_qkv_weights), such that q, k, v are contiguous blocks instead of
    being interleaved along the innermost axis. Mathematically identical.
    """
    q, k, v = self.Wqkv(x).unflatten(-1, (3, self.num_heads, -1)).unbind(-3)
    q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
    enc = encoding.transpose(-3, -2)  # broadcast over (N, H) instead of (H, N)
    q, k = modules.apply_rotary(enc, q), modules.apply_rotary(enc, k)
    message = self.out_proj(_attention_bnhd(q, k, v, fused))
    return x + self.ffn(torch.cat([x, self.message_norm(message)], -1))


def _cross_block_forward(self, x0, x1, fused):
    """CrossBlock.forward with token-major (B, N, H, D) layout."""
    qk0, qk1, v0, v1 = (
        t.unflatten(-1, (self.num_heads, -1))
        for t in (self.to_qk(x0), self.to_qk(x1), self.to_v(x0), self.to_v(x1))
    )
    qk0, qk1 = self.qk_norm(qk0).to(v0.dtype), self.qk_norm(qk1).to(v1.dtype)
    m0 = self.to_out(_attention_bnhd(qk0, qk1, v1, fused))
    m1 = self.to_out(_attention_bnhd(qk1, qk0, v0, fused))
    x0 = x0 + self.ffn(torch.cat([x0, self.message_norm(m0)], -1))
    x1 = x1 + self.ffn(torch.cat([x1, self.message_norm(m1)], -1))
    return x0, x1


def entmax15_window_readout_topk(logits, grid, hw, radius: int = 1, k: int = 512):
    """ops.entmax15_window_readout with the argmax taken from the top-k.

    The reference takes the maximum and argmax with logits.max(-1) and then
    computes the top-k of the shifted logits for the entmax threshold. ONNX
    Runtime runs the ArgMax/ReduceMax on slow cuDNN reductions, so reuse the
    (sorted) top-k instead: its first entry is the maximum. Identical results.
    """
    topv, topi = logits.topk(k, dim=-1)
    vmax, idx = topv[..., :1], topi[..., :1]
    y = (logits - vmax) / 2
    tau = ops.sorted_entmax15_threshold((topv - vmax) / 2)
    h, w = hw
    r = torch.arange(-radius, radius + 1, device=y.device)
    ny = (idx // w)[..., None] + r[:, None]
    nx = (idx % w)[..., None] + r[None, :]
    inside = ((ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)).flatten(-2)
    nidx = (ny.clamp(0, h - 1) * w + nx.clamp(0, w - 1)).flatten(-2)
    p = (y.gather(-1, nidx) - tau).clamp_min(0).square()
    position = (p[..., None] * grid[nidx]).sum(-2) / p.sum(-1, keepdim=True)
    return position, (p * inside).sum(-1)


def _match_head_forward(self, points_ni, cf_i, cf_j):
    """MatchHead.forward without chunking the queries (identical results)."""
    search = F.normalize(self.to_search(cf_j.flatten(2).mT), dim=-1)
    return self.match_points(points_ni, cf_i, cf_j, search)


@contextlib.contextmanager
def unchunked_heads():
    """Export the correspondence heads on all queries at once.

    The reference splits the queries into chunks of query_chunk_size to bound
    the memory of the logits, which depends on the number of keypoints and
    cannot be exported with a dynamic number of keypoints.
    """
    original = modules.MatchHead.forward
    modules.MatchHead.forward = _match_head_forward
    try:
        yield
    finally:
        modules.MatchHead.forward = original


@contextlib.contextmanager
def topk_readout():
    """Export the correspondence readout with entmax15_window_readout_topk."""
    original = modules.ops.entmax15_window_readout
    modules.ops.entmax15_window_readout = entmax15_window_readout_topk
    try:
        yield
    finally:
        modules.ops.entmax15_window_readout = original


def permute_qkv_weights(matcher: FlexGlue) -> None:
    """Reorder the fused qkv projection from (H, D, 3) to (3, H, D) in place."""
    for block in matcher.transformers:
        attn = block.self_attn
        h = attn.num_heads
        dim = attn.Wqkv.out_features
        perm = torch.arange(dim).view(h, dim // (3 * h), 3).permute(2, 0, 1)
        perm = perm.reshape(-1)
        with torch.no_grad():
            attn.Wqkv.weight.copy_(attn.Wqkv.weight[perm])
            attn.Wqkv.bias.copy_(attn.Wqkv.bias[perm])


@contextlib.contextmanager
def token_major_attention(matcher: FlexGlue, fused: bool):
    """Export the transformer with contiguous q/k/v and (B, N, H, D) layout.

    The reference implementation interleaves q, k and v along the innermost
    axis of the qkv projection and transposes to (B, H, N, D). Splitting along
    an innermost axis of size 3 is very slow in ONNX Runtime (Split kernel with
    inner size 1), and the transposes are unnecessary for the fused attention,
    which takes (B, N, H*D) inputs. Permutes the weights of `matcher` in place.
    """
    permute_qkv_weights(matcher)
    self_forward = modules.SelfBlock.forward
    cross_forward = modules.CrossBlock.forward
    modules.SelfBlock.forward = lambda self, x, encoding: _self_block_forward(
        self, x, encoding, fused
    )
    modules.CrossBlock.forward = lambda self, x0, x1: _cross_block_forward(
        self, x0, x1, fused
    )
    try:
        yield
    finally:
        modules.SelfBlock.forward = self_forward
        modules.CrossBlock.forward = cross_forward


class DescriptorONNX(nn.Module):
    """Dense features of an image already resized to the input resolution."""

    def __init__(self, descriptor: DINOv3Descriptor):
        super().__init__()
        self.descriptor = descriptor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        d = self.descriptor
        return d.head(d.backbone((image - d.mean) / d.std))


def pairwise_sq_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared distances (B, M, N) between points a (B, M, 2) and b (B, N, 2).

    Computed from the coordinate differences rather than with the GEMM
    expansion |a|^2 + |b|^2 - 2ab (FlexGlue's gemm_nn), which loses precision
    for pixel coordinates, all the more if fp32 GEMMs run in TF32.
    """
    (ax, ay), (bx, by) = a.unbind(-1), b.unbind(-1)
    return (ax[:, :, None] - bx[:, None]).square() + (
        ay[:, :, None] - by[:, None]
    ).square()


class SparseMatcherONNX(nn.Module):
    """FlexGlue.match(..., sparse=True) with runtime thresholds (B=1).

    Mirrors FlexGlue.decode_pair, keypoint_correspondences (without cycle
    readouts, as in sparse mode), gate_points and sparse_matches (gemm_nn)
    without the Python control flow that depends on the thresholds, early
    stopping, refinement or CUDA graphs.
    """

    def __init__(self, matcher: FlexGlue, fp16: bool = False):
        super().__init__()
        self.matcher = matcher
        self.n_layers = matcher.conf.n_layers
        self.remove_border_px = matcher.conf.remove_border_px
        # Run the transformer layers in float16, as FlexGlue(mp="float16").
        # The readout always runs in float32.
        self.fp16 = fp16
        if fp16:
            for block in matcher.transformers:
                block.half()

    def decode_pair(self, f0: torch.Tensor, f1: torch.Tensor) -> PairContext:
        m = self.matcher
        hw0, hw1 = f0.shape[-2:], f1.shape[-2:]
        enc0, enc1 = (
            m.posenc(ops.cell_centers(*hw, scale=(max(hw),) * 2, device=f0.device))
            for hw in (hw0, hw1)
        )
        x0, x1 = m.input_proj(f0.flatten(2).mT), m.input_proj(f1.flatten(2).mT)
        if self.fp16:
            x0, x1, enc0, enc1 = x0.half(), x1.half(), enc0.half(), enc1.half()
        for block in m.transformers[: self.n_layers]:
            x0, x1 = block(x0, x1, enc0, enc1)
        x0, x1 = x0.float(), x1.float()
        covisibility_score = m.covisibility_score(x0, x1, self.n_layers)
        cf = (x0.mT.unflatten(-1, hw0), x1.mT.unflatten(-1, hw1))
        return PairContext(self.n_layers, covisibility_score, cf)

    def pixel_scale(self, features: torch.Tensor, image_size: torch.Tensor):
        resized_side = self.matcher.stride * max(features.shape[-2:])
        return image_size.max(-1).values / resized_side

    def forward(
        self,
        features0: torch.Tensor,
        features1: torch.Tensor,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        image_size0: torch.Tensor,
        image_size1: torch.Tensor,
        filter_threshold: torch.Tensor,
        max_distance_px: torch.Tensor,
        min_covisibility: torch.Tensor,
    ):
        head = self.matcher.heads[self.n_layers]
        context = self.decode_pair(features0, features1)

        kpts = (kpts0, kpts1)
        sizes = (image_size0, image_size1)
        scales = (
            self.pixel_scale(features0, image_size0),
            self.pixel_scale(features1, image_size1),
        )
        is_covisible = context.covisibility_score >= min_covisibility  # (B,)

        # Correspondences and covisibility/border checks, as
        # FlexGlue.keypoint_correspondences(cycle=False) and gate_points.
        p2d_ij, confidence, valid = [], [], []
        for i, j in ((0, 1), (1, 0)):
            cf_i, cf_j = context.direction(i)
            kpts_ni = ops.normalize_keypoints(kpts[i], sizes[i])
            p2di_nj, confidence_i = head(kpts_ni, cf_i, cf_j)
            p2di_ij = ops.denormalize_keypoints(p2di_nj, sizes[j])
            border_i = self.remove_border_px * scales[i]
            border_j = self.remove_border_px * scales[j]
            valid_i = is_covisible[:, None]
            valid_i = valid_i & ops.inside_border(kpts[i], sizes[i], border_i)
            valid_i = valid_i & ops.inside_border(p2di_ij, sizes[j], border_j)
            p2d_ij.append(p2di_ij)
            confidence.append(confidence_i)
            valid.append(valid_i)

        # Mutual nearest neighbors on the summed squared distances, as
        # FlexGlue.sparse_matches with gemm_nn.
        sq_dist = pairwise_sq_distance(p2d_ij[0], kpts1) + pairwise_sq_distance(
            kpts0, p2d_ij[1]
        )
        m0, _ = ops.mutual_nn(sq_dist)
        m0_safe = m0.clamp(min=0)

        def take(t: torch.Tensor) -> torch.Tensor:
            return t.gather(1, m0_safe[..., None].expand(-1, -1, 2))

        # Exact mean distance of the selected pairs.
        dist0 = 0.5 * (
            (p2d_ij[0] - take(kpts1)).norm(dim=-1)
            + (kpts0 - take(p2d_ij[1])).norm(dim=-1)
        )
        # Geometric mean of both confidences.
        scores0 = (confidence[0] * confidence[1].gather(1, m0_safe)).sqrt()
        valid0 = (m0 > -1) & (scores0 >= filter_threshold)
        valid0 = valid0 & valid[0] & valid[1].gather(1, m0_safe)
        scale = (scales[0] + scales[1]) / 2
        valid0 = valid0 & (dist0 < max_distance_px * scale[:, None])
        matches0 = torch.where(valid0, m0, -1)
        mscores0 = torch.where(valid0, scores0, 0)
        return matches0, mscores0, context.covisibility_score


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_descriptor(descriptor, path: Path, size, opset: int):
    model = DescriptorONNX(descriptor).eval()
    image = torch.rand(1, 3, *size)
    torch.onnx.export(
        model,
        (image,),
        str(path),
        input_names=["image"],
        output_names=["features"],
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )


def dummy_matcher_inputs(size, n0=1000, n1=1200, device="cpu"):
    h, w = size[0] // 8, size[1] // 8
    image_size0 = torch.tensor([[1024.0, 768.0]], device=device)
    image_size1 = torch.tensor([[800.0, 1000.0]], device=device)
    return (
        torch.randn(1, 256, h, w, device=device),
        torch.randn(1, 256, h, w, device=device),
        torch.rand(1, n0, 2, device=device) * image_size0[:, None],
        torch.rand(1, n1, 2, device=device) * image_size1[:, None],
        image_size0,
        image_size1,
        torch.tensor(0.1, device=device),
        torch.tensor(3.0, device=device),
        torch.tensor(0.0, device=device),
    )


MATCHER_INPUTS = [
    "features0",
    "features1",
    "kpts0",
    "kpts1",
    "image_size0",
    "image_size1",
    "filter_threshold",
    "max_distance_px",
    "min_covisibility",
]
MATCHER_OUTPUTS = ["matches0", "mscores0", "covisibility_score"]


def make_keypoint_dims_dynamic(path: Path):
    """Remove shapes specialized to the dummy number of keypoints.

    The exporter may annotate intermediate values (value_info) and outputs with
    the concrete number of keypoints of the dummy inputs. ONNX Runtime uses
    these annotations in graph optimizations (e.g. MatMul+Add -> Gemm fusion
    with static reshapes), which then fail for other numbers of keypoints.
    Clear the intermediate annotations and name the dynamic dimensions.
    """
    import onnx

    model = onnx.load(str(path))
    del model.graph.value_info[:]
    dims = {
        "kpts0": "num_keypoints0",
        "kpts1": "num_keypoints1",
        "matches0": "num_keypoints0",
        "mscores0": "num_keypoints0",
    }
    for value in list(model.graph.input) + list(model.graph.output):
        if value.name in dims:
            dim = value.type.tensor_type.shape.dim[1]
            dim.Clear()
            dim.dim_param = dims[value.name]
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def export_matcher(
    matcher,
    path: Path,
    size,
    opset: int,
    fp16: bool = False,
    fused: bool = True,
):
    model = SparseMatcherONNX(matcher, fp16=fp16).eval()
    inputs = dummy_matcher_inputs(size)
    n0 = torch.export.Dim("num_keypoints0", min=2, max=65536)
    n1 = torch.export.Dim("num_keypoints1", min=2, max=65536)
    dynamic_shapes = {
        "features0": None,
        "features1": None,
        "kpts0": {1: n0},
        "kpts1": {1: n1},
        "image_size0": None,
        "image_size1": None,
        "filter_threshold": None,
        "max_distance_px": None,
        "min_covisibility": None,
    }
    with token_major_attention(matcher, fused), unchunked_heads(), topk_readout():
        torch.onnx.export(
            model,
            inputs,
            str(path),
            input_names=MATCHER_INPUTS,
            output_names=MATCHER_OUTPUTS,
            dynamic_shapes=dynamic_shapes,
            opset_version=opset,
            dynamo=True,
            external_data=False,
            custom_translation_table={
                torch.ops.flexglue.multi_head_attention.default: (
                    multi_head_attention_onnx
                )
            },
        )
    make_keypoint_dims_dynamic(path)


def sift_keypoints(path: str, num_keypoints: int) -> torch.Tensor:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=num_keypoints)
    kpts = sorted(sift.detect(gray, None), key=lambda k: -k.response)
    # OpenCV returns the same location once per dominant orientation. Remove
    # duplicates, since ties in the mutual nearest neighbor search are broken
    # arbitrarily and would make the comparison flaky.
    pts = np.unique(np.array([k.pt for k in kpts], dtype=np.float32), axis=0)
    order = np.random.default_rng(0).permutation(len(pts))[:num_keypoints]
    # OpenCV puts the center of the top-left pixel at (0, 0), FlexGlue the
    # top-left corner of the image.
    return torch.from_numpy(pts[order] + 0.5)[None]


def check(args, descriptor, matcher, size):
    """Compare ONNX Runtime and PyTorch on image pairs from the assets."""
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if not args.cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print(f"ONNX Runtime {ort.__version__}, providers: {providers}")
    sess_desc = ort.InferenceSession(
        str(args.output_dir / "flexglue_descriptor.onnx"), providers=providers
    )
    sess_match = ort.InferenceSession(
        str(args.output_dir / args.matcher_name), providers=providers
    )

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    descriptor = descriptor.to(device)
    matcher = matcher.to(device)
    assets = Path(__file__).resolve().parent.parent / "assets"
    pairs = [
        ("berlin1.jpg", "berlin2.jpg"),
        ("lamar_cab_1446.jpg", "lamar_cab_1450.jpg"),
    ]
    num_kpts = [int(n) for n in args.check_num_keypoints.split(",")]
    all_ok = True
    for name0, name1 in pairs:
        image0 = load_image(assets / name0)
        image1 = load_image(assets / name1)
        # Descriptor: identical resize in both runtimes.
        x0 = pp.resize_images(image0[None], size)
        x1 = pp.resize_images(image1[None], size)
        with torch.no_grad():
            f0_pt = descriptor({"image": image0[None].to(device)}, resize=size)
            f1_pt = descriptor({"image": image1[None].to(device)}, resize=size)
        f0_ort = sess_desc.run(None, {"image": x0.numpy()})[0]
        f1_ort = sess_desc.run(None, {"image": x1.numpy()})[0]
        feat_err = max(
            np.abs(f0_ort - f0_pt["features"].cpu().numpy()).max(),
            np.abs(f1_ort - f1_pt["features"].cpu().numpy()).max(),
        )
        print(f"[{name0} / {name1}] descriptor max abs error: {feat_err:.2e}")

        for n in num_kpts:
            k0 = sift_keypoints(str(assets / name0), n)
            k1 = sift_keypoints(str(assets / name1), n)
            for min_covisibility in (0.0, 0.8):
                data0 = {**f0_pt, "keypoints": k0.to(device)}
                data1 = {**f1_pt, "keypoints": k1.to(device)}
                with torch.no_grad():
                    pred = matcher.match(
                        data0,
                        data1,
                        sparse=True,
                        mp=None,
                        min_covisibility=min_covisibility,
                    )
                m0_pt = pred["matches0"][0].cpu().numpy()
                feeds = {
                    "features0": f0_pt["features"].cpu().numpy(),
                    "features1": f1_pt["features"].cpu().numpy(),
                    "kpts0": k0.numpy(),
                    "kpts1": k1.numpy(),
                    "image_size0": f0_pt["image_size"].cpu().numpy(),
                    "image_size1": f1_pt["image_size"].cpu().numpy(),
                    "filter_threshold": np.array(0.1, np.float32),
                    "max_distance_px": np.array(3.0, np.float32),
                    "min_covisibility": np.array(min_covisibility, np.float32),
                }
                for _ in range(2):  # warmup
                    sess_match.run(None, feeds)
                num_runs = 3 if args.cpu else 10
                t = time.time()
                for _ in range(num_runs):
                    m0_ort, s0_ort, score_ort = sess_match.run(None, feeds)
                dt = (time.time() - t) / num_runs
                m0_ort = m0_ort[0]
                valid_pt, valid_ort = m0_pt > -1, m0_ort > -1
                both = valid_pt & valid_ort
                same = (both & (m0_pt == m0_ort)).sum()
                union = (valid_pt | valid_ort).sum()
                agreement = same / max(union, 1)
                num_diff = union - same
                score_err = abs(score_ort[0] - pred["covisibility_score"].item())
                # Tolerate rare flips of matches at the decision boundaries.
                ok = num_diff <= max(2, 0.01 * union) and score_err < 1e-3
                all_ok &= bool(ok)
                print(
                    f"  N={n:5d} min_covis={min_covisibility:.1f}: "
                    f"matches pt={valid_pt.sum():5d} ort={valid_ort.sum():5d} "
                    f"agreement={agreement:.4f} diff={num_diff} "
                    f"covis pt={pred['covisibility_score'].item():.4f} "
                    f"ort={score_ort[0]:.4f} ort_time={dt * 1000:.0f}ms "
                    f"{'OK' if ok else 'FAIL'}"
                )
    print("CHECK", "PASSED" if all_ok else "FAILED")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS_URL, help="Path or URL.")
    parser.add_argument(
        "--output_dir", type=Path, default=Path(__file__).resolve().parent / "models"
    )
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check_num_keypoints", default="512,2048,4096")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run the transformer layers of the matcher in float16 (GPU only).",
    )
    parser.add_argument(
        "--no_fused_attention",
        action="store_true",
        help="Decompose the attention into standard ONNX operators.",
    )
    parser.add_argument(
        "--matcher_only",
        action="store_true",
        help="Only export the matcher (the descriptor must exist in output_dir).",
    )
    args = parser.parse_args()

    args.matcher_name = (
        "flexglue_matcher_fp16.onnx" if args.fp16 else "flexglue_matcher.onnx"
    )
    size = tuple(args.size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    descriptor = DINOv3Descriptor(weights=args.weights, resize=size).eval()
    matcher = FlexGlue(weights=args.weights, describe=False, mp=None).eval()

    if not args.skip_export:
        exports = [(args.matcher_name, export_matcher, matcher)]
        if not args.matcher_only:
            exports.insert(
                0, ("flexglue_descriptor.onnx", export_descriptor, descriptor)
            )
        for name, fn, model in exports:
            path = args.output_dir / name
            t = time.time()
            if fn is export_matcher:
                # Copy, since the fp16 export converts the layers in place.
                fn(
                    copy.deepcopy(model),
                    path,
                    size,
                    args.opset,
                    fp16=args.fp16,
                    fused=not args.no_fused_attention,
                )
            else:
                fn(model, path, size, args.opset)
            print(
                f"Exported {path} in {time.time() - t:.0f}s "
                f"({path.stat().st_size / 1e6:.1f}MB, sha256={sha256sum(path)})"
            )

    if args.check and not check(args, descriptor, matcher, size):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
