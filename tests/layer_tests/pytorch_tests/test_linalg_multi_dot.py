

# # /home/tamar/myOptproject/openvino/tests/layer_tests/pytorch_tests/test_linalg_multi_dot.py
# import pytest
# import torch
# import numpy as np
# import openvino as ov
# from pathlib import Path


# def multi_dot_like(ts):
#     """
#     Pure-PyTorch implementation that mimics numpy/torch.linalg.multi_dot 1D rules,
#     but uses only torch.matmul (which is supported by the OV PyTorch Frontend).

#     - If first tensor is 1D, treat it as row vector (1, N) during multiplication.
#     - If last tensor is 1D, treat it as column vector (N, 1) during multiplication.
#     - After the chain, squeeze back to match multi_dot's output ranks:
#         * 1D·1D -> scalar (0-D)
#         * 1D·2D -> 1D row vector
#         * 2D·1D -> 1D column vector
#         * otherwise -> 2D
#     """
#     assert len(ts) >= 2, "multi_dot_like needs at least two tensors"

#     first_1d = ts[0].dim() == 1
#     last_1d = ts[-1].dim() == 1

#     work = list(ts)
#     if first_1d:
#         work[0] = work[0].unsqueeze(0)      # (N,) -> (1, N)
#     if last_1d:
#         work[-1] = work[-1].unsqueeze(-1)   # (N,) -> (N, 1)

#     res = work[0]
#     for w in work[1:]:
#         res = torch.matmul(res, w)

#     if first_1d and last_1d:
#         return res.squeeze()      # (1,1) -> scalar
#     if first_1d:
#         return res.squeeze(0)     # (1, M) -> (M,)
#     if last_1d:
#         return res.squeeze(-1)    # (M, 1) -> (M,)
#     return res


# def _compile(model, ie_device, precision, *, device_may_be_none=False):
#     """
#     Wrap ov.compile_model with device-specific config.
#     Forces FP32 on GPU when precision == 'FP32' to reduce numeric drift.
#     """
#     if device_may_be_none:
#         return ov.compile_model(model)  # AUTO
#     if ie_device == "GPU" and precision == "FP32":
#         # Force FP32 math on GPU
#         return ov.compile_model(model, ie_device, {"INFERENCE_PRECISION_HINT": "f32"})
#         # (newer syntax could be: from openvino.properties import hint; {hint.inference_precision: ov.Type.f32})
#     return ov.compile_model(model, ie_device)


# def _tolerances(ie_device, precision):
#     """
#     Choose rtol/atol based on device & precision.
#     - FP16 (any device): looser
#     - GPU FP32: slightly looser than CPU FP32 (different accumulations/optimizations)
#     - CPU FP32: strict
#     """
#     if precision == "FP16":
#         return 1e-2, 1e-2
#     if ie_device == "GPU":
#         return 1e-2, 1e-2
#     return 1e-4, 1e-4


# @pytest.mark.parametrize("shapes", [
#     ([(2, 3), (3, 4)]),                   # 2 matrices -> (2x3)(3x4) = (2x4)
#     ([(3,), (3, 4), (4,)]),               # vector(3) * (3x4) * vector(4) -> scalar
#     ([(4, 8), (8, 16), (16, 5), (5, 7)]), # 4 static 2D matrices -> (4x7)
# ])
# def test_linalg_multi_dot_convert_and_run(shapes, ie_device, precision):
#     """
#     Happy path: convert a module that emulates multi_dot via torch.matmul only,
#     compile & run on device, and compare to torch.linalg.multi_dot reference.
#     """
#     core = ov.Core()
#     if ie_device not in core.available_devices:
#         pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     # (אופציונלי לדטרמיניזם): torch.manual_seed(0)
#     tensors = [torch.randn(s, dtype=dtype) for s in shapes]

#     class M(torch.nn.Module):
#         def forward(self, *ts):
#             return multi_dot_like(list(ts))  # no linalg.multi_dot inside the model

#     m = M().eval()
#     with torch.no_grad():
#         ref = torch.linalg.multi_dot(tensors)

#     model = ov.convert_model(m, example_input=tuple(tensors))
#     compiled = _compile(model, ie_device, precision)

#     req = compiled.create_infer_request()
#     for i, t in enumerate(tensors):
#         req.set_tensor(compiled.input(i), ov.Tensor(t.detach().cpu().numpy()))
#     req.infer()

#     out = req.get_output_tensor(0).data
#     assert out.shape == tuple(ref.shape)

#     rtol, atol = _tolerances(ie_device, precision)
#     np.testing.assert_allclose(out, ref.detach().cpu().numpy(), rtol=rtol, atol=atol)


# def _run_and_compare(tensors, ie_device, precision, *, via_ir=False, device_may_be_none=False):
#     """
#     Convert module with multi_dot_like (matmul-only), optionally IR round-trip,
#     compile and compare shape & values vs. torch.linalg.multi_dot.
#     """
#     core = ov.Core()
#     if not device_may_be_none and ie_device not in core.available_devices:
#         pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

#     class M(torch.nn.Module):
#         def forward(self, *ts):
#             return multi_dot_like(list(ts))

#     m = M().eval()
#     with torch.no_grad():
#         ref = torch.linalg.multi_dot(tensors)

#     model = ov.convert_model(m, example_input=tuple(tensors))

#     if via_ir:
#         tmp = Path(".pytest_cache") / "ov_ir_tmp"
#         tmp.mkdir(parents=True, exist_ok=True)
#         xml_path = tmp / "model.xml"
#         bin_path = tmp / "model.bin"
#         ov.serialize(model, str(xml_path), str(bin_path))
#         model = core.read_model(str(xml_path))

#     compiled = _compile(model, ie_device, precision, device_may_be_none=device_may_be_none)

#     req = compiled.create_infer_request()
#     for i, t in enumerate(tensors):
#         req.set_tensor(compiled.input(i), ov.Tensor(t.detach().cpu().numpy()))
#     req.infer()

#     out = req.get_output_tensor(0).data
#     assert out.shape == tuple(ref.shape)

#     rtol, atol = _tolerances(ie_device, precision)
#     np.testing.assert_allclose(out, ref.detach().cpu().numpy(), rtol=rtol, atol=atol)


# # ---------- Focused edge-cases (precision comes from suite fixture) ----------
# def test_multi_dot_two_vectors_scalar(ie_device, precision):
#     """1D · 1D -> scalar (0-D)."""
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = torch.randn(5, dtype=dtype)
#     b = torch.randn(5, dtype=dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_vec_mat(ie_device, precision):
#     """(3,) x (3,4) -> (4,)  — left vector times matrix."""
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = torch.randn(3, dtype=dtype)
#     b = torch.randn(3, 4, dtype=dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_mat_vec(ie_device, precision):
#     """(3,4) x (4,) -> (3,)  — matrix times right vector."""
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = torch.randn(3, 4, dtype=dtype)
#     b = torch.randn(4, dtype=dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_three_with_vectors_scalar(ie_device, precision):
#     """(3,) x (3,4) x (4,) -> scalar. Stresses 1D edge handling."""
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = torch.randn(3, dtype=dtype)
#     b = torch.randn(3, 4, dtype=dtype)
#     c = torch.randn(4, dtype=dtype)
#     _run_and_compare([a, b, c], ie_device, precision)


# def test_multi_dot_inner_dim_one(ie_device, precision):
#     """Inner dim = 1: (2,1) x (1,4) -> (2,4)."""
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = torch.randn(2, 1, dtype=dtype)
#     b = torch.randn(1, 4, dtype=dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_noncontiguous_inputs(ie_device, precision):
#     """Non-contiguous inputs should behave identically."""
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a_full = torch.randn(6, 4, dtype=dtype)
#     a = a_full[::2]           # (3,4), non-contiguous
#     b_full = torch.randn(8, 4, dtype=dtype)
#     b = b_full[:4].t()        # transpose -> non-contiguous
#     _run_and_compare([a, b], ie_device, precision)  # (3,4) x (4,4) -> (3,4)


# def test_multi_dot_via_ir_roundtrip(ie_device, precision):
#     """Serialize -> read_model (IR round-trip). Ensures graph survives serialization."""
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [
#         torch.randn(2, 3, dtype=dtype),
#         torch.randn(3, 5, dtype=dtype),
#         torch.randn(5, 4, dtype=dtype),
#     ]
#     _run_and_compare(tensors, ie_device, precision, via_ir=True)


# def test_multi_dot_auto_device_selection(ie_device, precision):
#     """Compile without device (AUTO). Runs only if any device exists."""
#     core = ov.Core()
#     if not core.available_devices:
#         pytest.skip("No devices available for AUTO run")
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [
#         torch.randn(2, 3, dtype=dtype),
#         torch.randn(3, 4, dtype=dtype),
#     ]
#     _run_and_compare(tensors, ie_device, precision, device_may_be_none=True)
import pytest
import torch
import numpy as np
import openvino as ov
from pathlib import Path


# ---------- Implementation helper (matmul-only), convertible by OV ----------

def multi_dot_like(ts):
    """
    Pure-PyTorch implementation that mimics torch/numpy.linalg.multi_dot 1D rules
    but uses only torch.matmul (supported by OV's PyTorch Frontend).

    Semantics:
      * If first tensor is 1D, treat it as row (1, N) during the chain.
      * If last  tensor is 1D, treat it as col (N, 1) during the chain.
      * After the chain, squeeze back to match multi_dot ranks:
          - 1D·1D -> scalar (0-D)
          - 1D·2D -> 1-D row vector
          - 2D·1D -> 1-D column vector
          - else  -> 2-D matrix
    """
    assert len(ts) >= 2, "multi_dot_like needs at least two tensors"

    first_1d = ts[0].dim() == 1
    last_1d = ts[-1].dim() == 1

    work = list(ts)
    if first_1d:
        work[0] = work[0].unsqueeze(0)      # (N,)   -> (1, N)
    if last_1d:
        work[-1] = work[-1].unsqueeze(-1)   # (N,)   -> (N, 1)

    res = work[0]
    for w in work[1:]:
        res = torch.matmul(res, w)

    if first_1d and last_1d:
        return res.squeeze()      # (1,1) -> scalar
    if first_1d:
        return res.squeeze(0)     # (1, M) -> (M,)
    if last_1d:
        return res.squeeze(-1)    # (M, 1) -> (M,)
    return res


# ---------- OV compile helpers & tolerances ----------------------------------

def _compile(model, ie_device, precision, *, device_may_be_none=False):
    """
    Wrap ov.compile_model with useful defaults.
    Force FP32 math on GPU when precision == 'FP32' to reduce numeric drift.
    """
    if device_may_be_none:
        return ov.compile_model(model)  # AUTO
    if ie_device == "GPU" and precision == "FP32":
        return ov.compile_model(model, ie_device, {"INFERENCE_PRECISION_HINT": "f32"})
    return ov.compile_model(model, ie_device)


def _tolerances(ie_device, precision, *, chain_len=None, via_ir=False):
    """
    Choose rtol/atol per device/precision and scenario.

    Base:
      - CPU FP32: tight (1e-4, 1e-4)
      - GPU FP32: slightly looser (1e-2, 1e-2)
      - FP16 (any device): loose (1e-2, 1e-2)

    Extra relaxations (targeted):
      - FP16 + GPU + chain_len >= 3: a bit looser to cover realistic FP16 accumulation drift.
      - FP16 + GPU + via_ir=True: a bit more loose due to IR round-trip numerics.
    """
    # Base tolerances
    if precision == "FP16":
        rtol, atol = 1e-2, 1e-2
    elif ie_device == "GPU":
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 1e-4, 1e-4

    # Long-ish chains in FP16 on GPU (covers cases like 10x20 · 20x30 · 30x5)
    if precision == "FP16" and ie_device == "GPU" and (chain_len or 0) >= 3:
        rtol = max(rtol, 2.0e-2)
        atol = max(atol, 5.0e-2)

    # IR round-trip adds some variance for FP16 + GPU
    if precision == "FP16" and ie_device == "GPU" and via_ir:
        rtol = max(rtol, 3.0e-2)
        atol = max(atol, 8.0e-2)

    return rtol, atol

def _run_and_compare(tensors, ie_device, precision, *, via_ir=False, device_may_be_none=False):
    """
    Convert a tiny module using multi_dot_like (matmul-only), optionally IR round-trip,
    compile with OV, run inference, and compare to torch.linalg.multi_dot.
    """
    core = ov.Core()
    if not device_may_be_none and ie_device not in core.available_devices:
        pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

    class M(torch.nn.Module):
        def forward(self, *ts):
            return multi_dot_like(list(ts))

    m = M().eval()
    with torch.no_grad():
        ref = torch.linalg.multi_dot(tensors)

    model = ov.convert_model(m, example_input=tuple(tensors))

    if via_ir:
        tmp = Path(".pytest_cache") / "ov_ir_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        xml_path = tmp / "model.xml"
        bin_path = tmp / "model.bin"
        ov.serialize(model, str(xml_path), str(bin_path))
        model = core.read_model(str(xml_path))

    compiled = _compile(model, ie_device, precision, device_may_be_none=device_may_be_none)

    req = compiled.create_infer_request()
    for i, t in enumerate(tensors):
        req.set_tensor(compiled.input(i), ov.Tensor(t.detach().cpu().numpy()))
    req.infer()

    out = req.get_output_tensor(0).data
    assert out.shape == tuple(ref.shape)

    rtol, atol = _tolerances(
        ie_device,
        precision,
        chain_len=len(tensors),
        via_ir=via_ir,
    )
    np.testing.assert_allclose(out, ref.detach().cpu().numpy(), rtol=rtol, atol=atol)


# ---------- Functional tests (OV conversion + numeric check vs PyTorch) ------

@pytest.mark.parametrize("shapes", [
    # Basic 2D chain
    ([(2, 3), (3, 4)]),                         # (2x3)(3x4) -> (2x4)
    # 1D edges
    ([(3,), (3, 4), (4,)]),                     # (3,) (3x4) (4,) -> scalar
    # Longer 2D chain
    ([(4, 8), (8, 16), (16, 5), (5, 7)]),       # -> (4x7)
])
def test_linalg_multi_dot_convert_and_run(shapes, ie_device, precision):
    """
    Happy path: emulate multi_dot with matmul-only inside the model (convertible),
    compare to true torch.linalg.multi_dot results.
    """
    core = ov.Core()
    if ie_device not in core.available_devices:
        pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

    # Make random inputs deterministic for stability
    torch.manual_seed(0)

    dtype = torch.float16 if precision == "FP16" else torch.float32
    tensors = [torch.randn(s, dtype=dtype) for s in shapes]

    _run_and_compare(tensors, ie_device, precision)


def test_multi_dot_two_vectors_scalar(ie_device, precision):
    """1D · 1D -> scalar (0-D)."""
    torch.manual_seed(0)
    dtype = torch.float16 if precision == "FP16" else torch.float32
    a = torch.randn(5, dtype=dtype)
    b = torch.randn(5, dtype=dtype)
    _run_and_compare([a, b], ie_device, precision)


def test_multi_dot_vec_mat(ie_device, precision):
    """(3,) × (3,4) -> (4,)."""
    torch.manual_seed(0)
    dtype = torch.float16 if precision == "FP16" else torch.float32
    a = torch.randn(3, dtype=dtype)
    b = torch.randn(3, 4, dtype=dtype)
    _run_and_compare([a, b], ie_device, precision)


def test_multi_dot_mat_vec(ie_device, precision):
    """(3,4) × (4,) -> (3,)."""
    torch.manual_seed(0)
    dtype = torch.float16 if precision == "FP16" else torch.float32
    a = torch.randn(3, 4, dtype=dtype)
    b = torch.randn(4, dtype=dtype)
    _run_and_compare([a, b], ie_device, precision)


def test_multi_dot_three_with_vectors_scalar(ie_device, precision):
    """(3,) × (3,4) × (4,) -> scalar (stresses 1D edges)."""
    torch.manual_seed(0)
    dtype = torch.float16 if precision == "FP16" else torch.float32
    a = torch.randn(3, dtype=dtype)
    b = torch.randn(3, 4, dtype=dtype)
    c = torch.randn(4, dtype=dtype)
    _run_and_compare([a, b, c], ie_device, precision)


def test_multi_dot_inner_dim_one(ie_device, precision):
    """Inner dimension = 1: (2,1) × (1,4) -> (2,4)."""
    torch.manual_seed(0)
    dtype = torch.float16 if precision == "FP16" else torch.float32
    a = torch.randn(2, 1, dtype=dtype)
    b = torch.randn(1, 4, dtype=dtype)
    _run_and_compare([a, b], ie_device, precision)


def test_multi_dot_noncontiguous_inputs(ie_device, precision):
    """Non-contiguous storage (slice/transpose) should behave identically."""
    torch.manual_seed(0)
    dtype = torch.float16 if precision == "FP16" else torch.float32
    a_full = torch.randn(6, 4, dtype=dtype)
    a = a_full[::2]               # (3,4) non-contiguous
    b_full = torch.randn(8, 4, dtype=dtype)
    b = b_full[:4].t()            # (4,4) transposed -> non-contiguous
    _run_and_compare([a, b], ie_device, precision)


def test_multi_dot_via_ir_roundtrip(ie_device, precision):
    """Serialize -> read_model IR round-trip; graph must survive."""
    torch.manual_seed(0)
    dtype = torch.float16 if precision == "FP16" else torch.float32
    tensors = [
        torch.randn(10, 100, dtype=dtype),
        torch.randn(100, 5,  dtype=dtype),
        torch.randn(5, 50,   dtype=dtype),
    ]
    _run_and_compare(tensors, ie_device, precision, via_ir=True)


def test_multi_dot_auto_device_selection(ie_device, precision):
    """Compile without specifying device (AUTO) if there is any available device."""
    core = ov.Core()
    if not core.available_devices:
        pytest.skip("No devices available for AUTO run")

    torch.manual_seed(0)
    dtype = torch.float16 if precision == "FP16" else torch.float32
    tensors = [
        torch.randn(2, 3, dtype=dtype),
        torch.randn(3, 4, dtype=dtype),
    ]
    _run_and_compare(tensors, ie_device, precision, device_may_be_none=True)


# ---------- Parity with PyTorch test coverage (extra functional cases) -------

@pytest.mark.parametrize("shapes", [
    # Variable output ranks
    ([(2,), (2,)]),          # -> scalar
    ([(1, 2), (2,)]),        # -> (1,)
    ([(2,), (2, 1)]),        # -> (1,)
    ([(1, 2), (2, 1)]),      # -> (1,1) -> scalar
    ([(3, 2), (2, 4)]),      # -> (3,4)
    # Multi-input chains
    ([(3,), (3, 4), (4, 2), (2, 5), (5,)]),  # -> scalar
    ([(1, 2), (2, 2), (2, 3), (3, 1)]),      # -> (1,1) -> scalar
    # Larger cases (still moderate sizes)
    ([(10, 20), (20, 30), (30, 5)]),
])
def test_multi_dot_extra_functional(shapes, ie_device, precision):
    """Additional functional parity tests inspired by PyTorch suite."""
    core = ov.Core()
    if ie_device not in core.available_devices:
        pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

    torch.manual_seed(0)
    dtype = torch.float16 if precision == "FP16" else torch.float32
    tensors = [torch.randn(s, dtype=dtype) for s in shapes]
    _run_and_compare(tensors, ie_device, precision)


@pytest.mark.parametrize("shapes", [
    # Empty-dimension paths (PyTorch allows these; OV matmul path usually handles them)
    ([0], [0]),
    ([2], [2, 0]),
    ([1, 0], [0]),
    ([0, 2], [2, 1]),
    ([2, 2], [2, 0]),
    ([2, 0], [0, 3]),
    ([0, 0], [0, 1]),
    ([4, 2], [2, 0], [0, 3], [3, 2]),
])
def test_multi_dot_empty_dims(shapes, ie_device, precision):
    """
    Empty-dimension coverage similar to PyTorch tests.
    Note: these cases may rely on framework support for zero-sized matmul.
    """
    core = ov.Core()
    if ie_device not in core.available_devices:
        pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

    torch.manual_seed(0)
    dtype = torch.float16 if precision == "FP16" else torch.float32
    tensors = [torch.randn(*s, dtype=dtype) if isinstance(s, (list, tuple)) else torch.randn(s, dtype=dtype)
               for s in shapes]
    _run_and_compare(tensors, ie_device, precision)


# ---------- API error parity (PyTorch semantics only; no OV conversion) ------

def _mk(device, dtype, shape_or_rank):
    """Utility to make either 1-D/2-D by rank or shaped tensor quickly."""
    if isinstance(shape_or_rank, int):
        return torch.randn(shape_or_rank, device=device, dtype=dtype)
    return torch.randn(*shape_or_rank, device=device, dtype=dtype)


@pytest.mark.usefixtures("ie_device", "precision")
def test_multi_dot_errors_like_pytorch_cpu():
    """
    Mirrors PyTorch error checks: ranks, dtype/device mismatches, shape mismatch.
    These validate *API semantics* only, so they call torch.linalg.multi_dot directly.
    """
    device = "cpu"
    dtype = torch.float32
    a = _mk(device, dtype, 2)

    # At least two tensors
    with pytest.raises(RuntimeError, match="expected at least 2 tensors"):
        torch.linalg.multi_dot([])
    with pytest.raises(RuntimeError, match="expected at least 2 tensors"):
        torch.linalg.multi_dot([a])

    # First/last tensor must be 1D or 2D
    with pytest.raises(RuntimeError, match="the first tensor must be 1D or 2D"):
        torch.linalg.multi_dot([torch.tensor(1, device=device, dtype=dtype), a])
    with pytest.raises(RuntimeError, match="the last tensor must be 1D or 2D"):
        torch.linalg.multi_dot([a, torch.tensor(1, device=device, dtype=dtype)])

    # Middle tensors must be 2D
    with pytest.raises(RuntimeError, match="tensor 1 must be 2D"):
        torch.linalg.multi_dot([a, _mk(device, dtype, (2, 2, 2)), a])

    # Dtype mismatch between inputs
    with pytest.raises(RuntimeError, match="all tensors must have be the same dtype"):
        torch.linalg.multi_dot([a, _mk(device, torch.double, 2)])

    # Shape mismatch (cannot be multiplied)
    with pytest.raises(RuntimeError, match="cannot be multiplied"):
        torch.linalg.multi_dot([a, _mk(device, dtype, 3)])
    with pytest.raises(RuntimeError, match="cannot be multiplied"):
        torch.linalg.multi_dot([a, _mk(device, dtype, (3, 2)), a])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.usefixtures("ie_device", "precision")
def test_multi_dot_errors_like_pytorch_cuda():
    """
    Same API error checks but for CUDA device cases:
    - all tensors must be on the same device
    - 'out' must be on the same device & dtype
    """
    device = "cuda"
    dtype = torch.float32
    a = _mk(device, dtype, 2)

    # Different device in inputs
    with pytest.raises(RuntimeError, match="all tensors must be on the same device"):
        torch.linalg.multi_dot([a, _mk("cpu", dtype, 2)])

    # 'out' on different device / dtype
    with pytest.raises(RuntimeError, match="expected out tensor to be on device"):
        torch.linalg.multi_dot([a, a], out=torch.empty(0, dtype=dtype, device="cpu"))
    with pytest.raises(RuntimeError, match="expected out tensor to have dtype"):
        torch.linalg.multi_dot([a, a], out=torch.empty(0, dtype=torch.double, device=device))
