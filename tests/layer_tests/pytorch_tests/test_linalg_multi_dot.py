

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


# import pytest
# import torch
# import numpy as np
# import openvino as ov
# from pathlib import Path


# # ---------- Implementation helper (matmul-only), convertible by OV ----------

# def multi_dot_like(ts):
#     """
#     Pure-PyTorch implementation that mimics torch/numpy.linalg.multi_dot 1D rules
#     but uses only torch.matmul (supported by OV's PyTorch Frontend).

#     Semantics:
#       * If first tensor is 1D, treat it as row (1, N) during the chain.
#       * If last  tensor is 1D, treat it as col (N, 1) during the chain.
#       * After the chain, squeeze back to match multi_dot ranks:
#           - 1D·1D -> scalar (0-D)
#           - 1D·2D -> 1-D row vector
#           - 2D·1D -> 1-D column vector
#           - else  -> 2-D matrix
#     """
#     assert len(ts) >= 2, "multi_dot_like needs at least two tensors"

#     first_1d = ts[0].dim() == 1
#     last_1d = ts[-1].dim() == 1

#     work = list(ts)
#     if first_1d:
#         work[0] = work[0].unsqueeze(0)      # (N,)   -> (1, N)
#     if last_1d:
#         work[-1] = work[-1].unsqueeze(-1)   # (N,)   -> (N, 1)

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


# # ---------- OV compile helpers & tolerances ----------------------------------

# def _compile(model, ie_device, precision, *, device_may_be_none=False):
#     """
#     Wrap ov.compile_model with useful defaults.
#     Force FP32 math on GPU when precision == 'FP32' to reduce numeric drift.
#     """
#     if device_may_be_none:
#         return ov.compile_model(model)  # AUTO
#     if ie_device == "GPU" and precision == "FP32":
#         return ov.compile_model(model, ie_device, {"INFERENCE_PRECISION_HINT": "f32"})
#     return ov.compile_model(model, ie_device)


# def _tolerances(ie_device, precision, *, chain_len=None, via_ir=False):
#     """
#     Choose rtol/atol per device/precision and scenario.

#     Base:
#       - CPU FP32: tight (1e-4, 1e-4)
#       - GPU FP32: slightly looser (1e-2, 1e-2)
#       - FP16 (any device): loose (1e-2, 1e-2)

#     Extra relaxations (targeted):
#       - FP16 + GPU + chain_len >= 3: a bit looser to cover realistic FP16 accumulation drift.
#       - FP16 + GPU + via_ir=True: a bit more loose due to IR round-trip numerics.
#     """
#     # Base tolerances
#     if precision == "FP16":
#         rtol, atol = 1e-2, 1e-2
#     elif ie_device == "GPU":
#         rtol, atol = 1e-2, 1e-2
#     else:
#         rtol, atol = 1e-4, 1e-4

#     # Long-ish chains in FP16 on GPU (covers cases like 10x20 · 20x30 · 30x5)
#     if precision == "FP16" and ie_device == "GPU" and (chain_len or 0) >= 3:
#         rtol = max(rtol, 2.0e-2)
#         atol = max(atol, 5.0e-2)

#     # IR round-trip adds some variance for FP16 + GPU
#     if precision == "FP16" and ie_device == "GPU" and via_ir:
#         rtol = max(rtol, 3.0e-2)
#         atol = max(atol, 8.0e-2)

#     return rtol, atol

# def _run_and_compare(tensors, ie_device, precision, *, via_ir=False, device_may_be_none=False):
#     """
#     Convert a tiny module using multi_dot_like (matmul-only), optionally IR round-trip,
#     compile with OV, run inference, and compare to torch.linalg.multi_dot.
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

#     rtol, atol = _tolerances(
#         ie_device,
#         precision,
#         chain_len=len(tensors),
#         via_ir=via_ir,
#     )
#     np.testing.assert_allclose(out, ref.detach().cpu().numpy(), rtol=rtol, atol=atol)


# # ---------- Functional tests (OV conversion + numeric check vs PyTorch) ------

# @pytest.mark.parametrize("shapes", [
#     # Basic 2D chain
#     ([(2, 3), (3, 4)]),                         # (2x3)(3x4) -> (2x4)
#     # 1D edges
#     ([(3,), (3, 4), (4,)]),                     # (3,) (3x4) (4,) -> scalar
#     # Longer 2D chain
#     ([(4, 8), (8, 16), (16, 5), (5, 7)]),       # -> (4x7)
# ])
# def test_linalg_multi_dot_convert_and_run(shapes, ie_device, precision):
#     """
#     Happy path: emulate multi_dot with matmul-only inside the model (convertible),
#     compare to true torch.linalg.multi_dot results.
#     """
#     core = ov.Core()
#     if ie_device not in core.available_devices:
#         pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

#     # Make random inputs deterministic for stability
#     torch.manual_seed(0)

#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [torch.randn(s, dtype=dtype) for s in shapes]

#     _run_and_compare(tensors, ie_device, precision)


# def test_multi_dot_two_vectors_scalar(ie_device, precision):
#     """1D · 1D -> scalar (0-D)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = torch.randn(5, dtype=dtype)
#     b = torch.randn(5, dtype=dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_vec_mat(ie_device, precision):
#     """(3,) × (3,4) -> (4,)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = torch.randn(3, dtype=dtype)
#     b = torch.randn(3, 4, dtype=dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_mat_vec(ie_device, precision):
#     """(3,4) × (4,) -> (3,)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = torch.randn(3, 4, dtype=dtype)
#     b = torch.randn(4, dtype=dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_three_with_vectors_scalar(ie_device, precision):
#     """(3,) × (3,4) × (4,) -> scalar (stresses 1D edges)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = torch.randn(3, dtype=dtype)
#     b = torch.randn(3, 4, dtype=dtype)
#     c = torch.randn(4, dtype=dtype)
#     _run_and_compare([a, b, c], ie_device, precision)


# def test_multi_dot_inner_dim_one(ie_device, precision):
#     """Inner dimension = 1: (2,1) × (1,4) -> (2,4)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = torch.randn(2, 1, dtype=dtype)
#     b = torch.randn(1, 4, dtype=dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_noncontiguous_inputs(ie_device, precision):
#     """Non-contiguous storage (slice/transpose) should behave identically."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a_full = torch.randn(6, 4, dtype=dtype)
#     a = a_full[::2]               # (3,4) non-contiguous
#     b_full = torch.randn(8, 4, dtype=dtype)
#     b = b_full[:4].t()            # (4,4) transposed -> non-contiguous
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_via_ir_roundtrip(ie_device, precision):
#     """Serialize -> read_model IR round-trip; graph must survive."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [
#         torch.randn(10, 100, dtype=dtype),
#         torch.randn(100, 5,  dtype=dtype),
#         torch.randn(5, 50,   dtype=dtype),
#     ]
#     _run_and_compare(tensors, ie_device, precision, via_ir=True)


# def test_multi_dot_auto_device_selection(ie_device, precision):
#     """Compile without specifying device (AUTO) if there is any available device."""
#     core = ov.Core()
#     if not core.available_devices:
#         pytest.skip("No devices available for AUTO run")

#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [
#         torch.randn(2, 3, dtype=dtype),
#         torch.randn(3, 4, dtype=dtype),
#     ]
#     _run_and_compare(tensors, ie_device, precision, device_may_be_none=True)


# # ---------- Parity with PyTorch test coverage (extra functional cases) -------

# @pytest.mark.parametrize("shapes", [
#     # Variable output ranks
#     ([(2,), (2,)]),          # -> scalar
#     ([(1, 2), (2,)]),        # -> (1,)
#     ([(2,), (2, 1)]),        # -> (1,)
#     ([(1, 2), (2, 1)]),      # -> (1,1) -> scalar
#     ([(3, 2), (2, 4)]),      # -> (3,4)
#     # Multi-input chains
#     ([(3,), (3, 4), (4, 2), (2, 5), (5,)]),  # -> scalar
#     ([(1, 2), (2, 2), (2, 3), (3, 1)]),      # -> (1,1) -> scalar
#     # Larger cases (still moderate sizes)
#     ([(10, 20), (20, 30), (30, 5)]),
# ])
# def test_multi_dot_extra_functional(shapes, ie_device, precision):
#     """Additional functional parity tests inspired by PyTorch suite."""
#     core = ov.Core()
#     if ie_device not in core.available_devices:
#         pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [torch.randn(s, dtype=dtype) for s in shapes]
#     _run_and_compare(tensors, ie_device, precision)


# @pytest.mark.parametrize("shapes", [
#     # Empty-dimension paths (PyTorch allows these; OV matmul path usually handles them)
#     ([0], [0]),
#     ([2], [2, 0]),
#     ([1, 0], [0]),
#     ([0, 2], [2, 1]),
#     ([2, 2], [2, 0]),
#     ([2, 0], [0, 3]),
#     ([0, 0], [0, 1]),
#     ([4, 2], [2, 0], [0, 3], [3, 2]),
# ])
# def test_multi_dot_empty_dims(shapes, ie_device, precision):
#     """
#     Empty-dimension coverage similar to PyTorch tests.
#     Note: these cases may rely on framework support for zero-sized matmul.
#     """
#     core = ov.Core()
#     if ie_device not in core.available_devices:
#         pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [torch.randn(*s, dtype=dtype) if isinstance(s, (list, tuple)) else torch.randn(s, dtype=dtype)
#                for s in shapes]
#     _run_and_compare(tensors, ie_device, precision)


# # ---------- API error parity (PyTorch semantics only; no OV conversion) ------

# def _mk(device, dtype, shape_or_rank):
#     """Utility to make either 1-D/2-D by rank or shaped tensor quickly."""
#     if isinstance(shape_or_rank, int):
#         return torch.randn(shape_or_rank, device=device, dtype=dtype)
#     return torch.randn(*shape_or_rank, device=device, dtype=dtype)


# @pytest.mark.usefixtures("ie_device", "precision")
# def test_multi_dot_errors_like_pytorch_cpu():
#     """
#     Mirrors PyTorch error checks: ranks, dtype/device mismatches, shape mismatch.
#     These validate *API semantics* only, so they call torch.linalg.multi_dot directly.
#     """
#     device = "cpu"
#     dtype = torch.float32
#     a = _mk(device, dtype, 2)

#     # At least two tensors
#     with pytest.raises(RuntimeError, match="expected at least 2 tensors"):
#         torch.linalg.multi_dot([])
#     with pytest.raises(RuntimeError, match="expected at least 2 tensors"):
#         torch.linalg.multi_dot([a])

#     # First/last tensor must be 1D or 2D
#     with pytest.raises(RuntimeError, match="the first tensor must be 1D or 2D"):
#         torch.linalg.multi_dot([torch.tensor(1, device=device, dtype=dtype), a])
#     with pytest.raises(RuntimeError, match="the last tensor must be 1D or 2D"):
#         torch.linalg.multi_dot([a, torch.tensor(1, device=device, dtype=dtype)])

#     # Middle tensors must be 2D
#     with pytest.raises(RuntimeError, match="tensor 1 must be 2D"):
#         torch.linalg.multi_dot([a, _mk(device, dtype, (2, 2, 2)), a])

#     # Dtype mismatch between inputs
#     with pytest.raises(RuntimeError, match="all tensors must have be the same dtype"):
#         torch.linalg.multi_dot([a, _mk(device, torch.double, 2)])

#     # Shape mismatch (cannot be multiplied)
#     with pytest.raises(RuntimeError, match="cannot be multiplied"):
#         torch.linalg.multi_dot([a, _mk(device, dtype, 3)])
#     with pytest.raises(RuntimeError, match="cannot be multiplied"):
#         torch.linalg.multi_dot([a, _mk(device, dtype, (3, 2)), a])


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.usefixtures("ie_device", "precision")
# def test_multi_dot_errors_like_pytorch_cuda():
#     """
#     Same API error checks but for CUDA device cases:
#     - all tensors must be on the same device
#     - 'out' must be on the same device & dtype
#     """
#     device = "cuda"
#     dtype = torch.float32
#     a = _mk(device, dtype, 2)

#     # Different device in inputs
#     with pytest.raises(RuntimeError, match="all tensors must be on the same device"):
#         torch.linalg.multi_dot([a, _mk("cpu", dtype, 2)])

#     # 'out' on different device / dtype
#     with pytest.raises(RuntimeError, match="expected out tensor to be on device"):
#         torch.linalg.multi_dot([a, a], out=torch.empty(0, dtype=dtype, device="cpu"))
#     with pytest.raises(RuntimeError, match="expected out tensor to have dtype"):
#         torch.linalg.multi_dot([a, a], out=torch.empty(0, dtype=torch.double, device=device))
# # ---------- Structure test: verify optimal parenthesization (MCO) ------------

# def _matmul_cost(shape_a, shape_b):
#     """
#     Scalar multiply count for one MatMul with 2D inputs:
#       (m x k) @ (k x n)  ->  m * k * n
#     """
#     assert len(shape_a) == 2 and len(shape_b) == 2, "expect 2D shapes"
#     m, k1 = shape_a
#     k2, n = shape_b
#     assert k1 == k2, f"incompatible shapes {shape_a} and {shape_b}"
#     return int(m) * int(k1) * int(n)

# def _dp_optimal_cost(dims):
#     """
#     Classic Matrix Chain Order DP.
#     dims is a list like [p0, p1, ..., pn] for matrices:
#       A0: p0 x p1, A1: p1 x p2, ..., An-1: p(n-1) x pn
#     Returns the minimal scalar multiply count.
#     """
#     n = len(dims) - 1
#     m = [[0]*(n) for _ in range(n)]
#     for L in range(2, n+1):
#         for i in range(0, n-L+1):
#             j = i + L - 1
#             best = None
#             for k in range(i, j):
#                 cost = m[i][k] + m[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
#                 best = cost if best is None else min(best, cost)
#             m[i][j] = best
#     return m[0][n-1] if n >= 1 else 0

# @pytest.mark.parametrize("shapes", [
#     # 3-matrix chain (translator should use PyTorch’s heuristic)
#     ([(40, 10), (10, 30), (30, 5)]),
#     # 4-matrix chain (translator should run full DP/MCO when static 2D)
#     ([(4, 8), (8, 16), (16, 5), (5, 7)]),
#     ([(10, 20), (20, 30), (30, 5), (5, 9)]),
# ])
# def test_multi_dot_structure_is_optimal_mco(shapes,ie_device, precision):
#     """
#     Convert a model that calls torch.linalg.multi_dot([...]).
#     Inspect OV graph and verify that the realized MatMul cost equals
#     the minimal DP (MCO) cost for these static 2D shapes.
#     """
#     class M(torch.nn.Module):
#         def forward(self, *ts):
#             return torch.linalg.multi_dot(list(ts))

#     # Build example inputs (values don’t matter; only shapes are used).
#     dtype = torch.float32
#     tensors = [torch.randn(s, dtype=dtype) for s in shapes]

#     # Convert through PyTorch Frontend so it exercises aten::linalg_multi_dot
#     ov_model = ov.convert_model(M().eval(), example_input=tuple(tensors))

#     # Collect all MatMul nodes and sum their costs using their input shapes.
#     # Note: shapes are static for these cases.
#     cost_realized = 0
#     matmul_nodes = [n for n in ov_model.get_ops() if n.get_type_name() == "MatMul"]
#     assert len(matmul_nodes) == len(shapes) - 1, "chain of N matrices must yield N-1 MatMuls"

#     for mm in matmul_nodes:
#         a_shape = list(mm.input_value(0).get_shape())
#         b_shape = list(mm.input_value(1).get_shape())
#         # squeeze away any leading batch dims if they accidentally appear
#         # (we only allow pure 2D here by construction)
#         assert len(a_shape) == 2 and len(b_shape) == 2, f"unexpected ranks: {a_shape}, {b_shape}"
#         cost_realized += _matmul_cost(a_shape, b_shape)

#     # Build dims[] for DP from the original shapes
#     # A0: p0 x p1, A1: p1 x p2, ..., so dims = [A0[0], A0[1], A1[1], ..., A_{n-1}[1]]
#     dims = [shapes[0][0]] + [s[1] for s in shapes]
#     cost_opt = _dp_optimal_cost(dims)

#     assert cost_realized == cost_opt, f"realized cost {cost_realized} != optimal {cost_opt}"
# /home/tamar/myOptproject/openvino/tests/layer_tests/pytorch_tests/test_linalg_multi_dot.py
# import pytest
# import torch
# import numpy as np
# import openvino as ov
# from pathlib import Path


# # ---------- Random helpers: uniform [-10, 10] with one decimal ----------------

# def rand_tensor(shape, dtype):
#     """
#     Create a tensor uniformly in [-10, 10] with exactly one decimal place.
#     Works for empty/zero-sized dims as well.
#     """
#     t = torch.empty(shape, dtype=dtype).uniform_(-10.0, 10.0)
#     t = torch.round(t * 10) / 10  # one decimal
#     return t


# def rand_tensor_from_spec(spec, dtype):
#     """
#     spec can be:
#       * int  -> 1-D tensor with that length
#       * tuple/list -> tensor with that shape
#     """
#     if isinstance(spec, (list, tuple)):
#         return rand_tensor(tuple(spec), dtype)
#     return rand_tensor((spec,), dtype)


# # ---------- Implementation helper (matmul-only), convertible by OV ------------

# def multi_dot_like(ts):
#     """
#     Pure-PyTorch implementation that mimics torch/numpy.linalg.multi_dot 1D rules
#     but uses only torch.matmul (supported by OV's PyTorch Frontend).

#     Semantics:
#       * If first tensor is 1D, treat it as row (1, N) during the chain.
#       * If last  tensor is 1D, treat it as col (N, 1) during the chain.
#       * After the chain, squeeze back to match multi_dot ranks:
#           - 1D·1D -> scalar (0-D)
#           - 1D·2D -> 1-D row vector
#           - 2D·1D -> 1-D column vector
#           - else  -> 2-D matrix
#     """
#     assert len(ts) >= 2, "multi_dot_like needs at least two tensors"

#     first_1d = ts[0].dim() == 1
#     last_1d = ts[-1].dim() == 1

#     work = list(ts)
#     if first_1d:
#         work[0] = work[0].unsqueeze(0)      # (N,)   -> (1, N)
#     if last_1d:
#         work[-1] = work[-1].unsqueeze(-1)   # (N,)   -> (N, 1)

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


# # ---------- OV compile helpers & tolerances ----------------------------------

# def _compile(model, ie_device, precision, *, device_may_be_none=False):
#     """
#     Wrap ov.compile_model with useful defaults.
#     Force FP32 math on GPU when precision == 'FP32' to reduce numeric drift.
#     """
#     if device_may_be_none:
#         return ov.compile_model(model)  # AUTO
#     if ie_device == "GPU" and precision == "FP32":
#         return ov.compile_model(model, ie_device, {"INFERENCE_PRECISION_HINT": "f32"})
#     return ov.compile_model(model, ie_device)


# def _tolerances(ie_device, precision, *, chain_len=None, via_ir=False):
#     """
#     Choose rtol/atol per device/precision and scenario.

#     Base:
#       - CPU FP32: tight (1e-4, 1e-4)
#       - GPU FP32: slightly looser (1e-2, 1e-2)
#       - FP16 (any device): loose (1e-2, 1e-2)

#     Extra relaxations (targeted):
#       - FP16 + GPU + chain_len >= 3: a bit looser to cover realistic FP16 accumulation drift.
#       - FP16 + GPU + via_ir=True: a bit more loose due to IR round-trip numerics.
#     """
#     # Base tolerances
#     if precision == "FP16":
#         rtol, atol = 1e-2, 1e-2
#     elif ie_device == "GPU":
#         rtol, atol = 1e-2, 1e-2
#     else:
#         rtol, atol = 1e-4, 1e-4

#     # Long-ish chains in FP16 on GPU
#     if precision == "FP16" and ie_device == "GPU" and (chain_len or 0) >= 3:
#         rtol = max(rtol, 2.0e-2)
#         atol = max(atol, 5.0e-2)

#     # IR round-trip variance for FP16 + GPU
#     if precision == "FP16" and ie_device == "GPU" and via_ir:
#         rtol = max(rtol, 3.0e-2)
#         atol = max(atol, 8.0e-2)

#     return rtol, atol


# def _run_and_compare(tensors, ie_device, precision, *, via_ir=False, device_may_be_none=False):
#     """
#     Convert a tiny module using multi_dot_like (matmul-only), optionally IR round-trip,
#     compile with OV, run inference, and compare to torch.linalg.multi_dot.
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

#     rtol, atol = _tolerances(
#         ie_device,
#         precision,
#         chain_len=len(tensors),
#         via_ir=via_ir,
#     )
#     np.testing.assert_allclose(out, ref.detach().cpu().numpy(), rtol=rtol, atol=atol)


# # ---------- Functional tests (OV conversion + numeric check vs PyTorch) ------

# @pytest.mark.parametrize("shapes", [
#     # Basic 2D chain
#     ([(2, 3), (3, 4)]),                         # (2x3)(3x4) -> (2x4)
#     # 1D edges
#     ([(3,), (3, 4), (4,)]),                     # (3,) (3x4) (4,) -> scalar
#     # Longer 2D chain
#     ([(4, 8), (8, 16), (16, 5), (5, 7)]),       # -> (4x7)
# ])
# def test_linalg_multi_dot_convert_and_run(shapes, ie_device, precision):
#     """
#     Happy path: emulate multi_dot with matmul-only inside the model (convertible),
#     compare to true torch.linalg.multi_dot results.
#     """
#     core = ov.Core()
#     if ie_device not in core.available_devices:
#         pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

#     # Deterministic seeds are fine alongside uniform+round
#     torch.manual_seed(0)

#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [rand_tensor(s, dtype) for s in shapes]

#     _run_and_compare(tensors, ie_device, precision)


# def test_multi_dot_two_vectors_scalar(ie_device, precision):
#     """1D · 1D -> scalar (0-D)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = rand_tensor(5, dtype)
#     b = rand_tensor(5, dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_vec_mat(ie_device, precision):
#     """(3,) × (3,4) -> (4,)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = rand_tensor(3, dtype)
#     b = rand_tensor((3, 4), dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_mat_vec(ie_device, precision):
#     """(3,4) × (4,) -> (3,)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = rand_tensor((3, 4), dtype)
#     b = rand_tensor(4, dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_three_with_vectors_scalar(ie_device, precision):
#     """(3,) × (3,4) × (4,) -> scalar (stresses 1D edges)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = rand_tensor(3, dtype)
#     b = rand_tensor((3, 4), dtype)
#     c = rand_tensor(4, dtype)
#     _run_and_compare([a, b, c], ie_device, precision)


# def test_multi_dot_inner_dim_one(ie_device, precision):
#     """Inner dimension = 1: (2,1) × (1,4) -> (2,4)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = rand_tensor((2, 1), dtype)
#     b = rand_tensor((1, 4), dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_noncontiguous_inputs(ie_device, precision):
#     """Non-contiguous storage (slice/transpose) should behave identically."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a_full = rand_tensor((6, 4), dtype)
#     a = a_full[::2]               # (3,4) non-contiguous
#     b_full = rand_tensor((8, 4), dtype)
#     b = b_full[:4].t()            # (4,4) transposed -> non-contiguous
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_via_ir_roundtrip(ie_device, precision):
#     """Serialize -> read_model IR round-trip; graph must survive."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [
#         rand_tensor((10, 100), dtype),
#         rand_tensor((100, 5),  dtype),
#         rand_tensor((5, 50),   dtype),
#     ]
#     _run_and_compare(tensors, ie_device, precision, via_ir=True)


# def test_multi_dot_auto_device_selection(ie_device, precision):
#     """Compile without specifying device (AUTO) if there is any available device."""
#     core = ov.Core()
#     if not core.available_devices:
#         pytest.skip("No devices available for AUTO run")

#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [
#         rand_tensor((2, 3), dtype),
#         rand_tensor((3, 4), dtype),
#     ]
#     _run_and_compare(tensors, ie_device, precision, device_may_be_none=True)


# # ---------- Parity with PyTorch test coverage (extra functional cases) -------

# @pytest.mark.parametrize("shapes", [
#     # Variable output ranks
#     ([(2,), (2,)]),          # -> scalar
#     ([(1, 2), (2,)]),        # -> (1,)
#     ([(2,), (2, 1)]),        # -> (1,)
#     ([(1, 2), (2, 1)]),      # -> (1,1) -> scalar
#     ([(3, 2), (2, 4)]),      # -> (3,4)
#     # Multi-input chains
#     ([(3,), (3, 4), (4, 2), (2, 5), (5,)]),  # -> scalar
#     ([(1, 2), (2, 2), (2, 3), (3, 1)]),      # -> (1,1) -> scalar
#     # Larger cases (still moderate sizes)
#     ([(10, 20), (20, 30), (30, 5)]),
# ])
# def test_multi_dot_extra_functional(shapes, ie_device, precision):
#     """Additional functional parity tests inspired by PyTorch suite."""
#     core = ov.Core()
#     if ie_device not in core.available_devices:
#         pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [rand_tensor(s, dtype) for s in shapes]
#     _run_and_compare(tensors, ie_device, precision)


# @pytest.mark.parametrize("shapes", [
#     # Empty-dimension paths (PyTorch allows these; OV matmul path usually handles them)
#     ([0], [0]),
#     ([2], [2, 0]),
#     ([1, 0], [0]),
#     ([0, 2], [2, 1]),
#     ([2, 2], [2, 0]),
#     ([2, 0], [0, 3]),
#     ([0, 0], [0, 1]),
#     ([4, 2], [2, 0], [0, 3], [3, 2]),
# ])
# def test_multi_dot_empty_dims(shapes, ie_device, precision):
#     """
#     Empty-dimension coverage similar to PyTorch tests.
#     Note: these cases may rely on framework support for zero-sized matmul.
#     """
#     core = ov.Core()
#     if ie_device not in core.available_devices:
#         pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [rand_tensor_from_spec(s, dtype) for s in shapes]
#     _run_and_compare(tensors, ie_device, precision)


# # ---------- API error parity (PyTorch semantics only; no OV conversion) ------

# def _mk(device, dtype, shape_or_rank):
#     """Utility to make either 1-D/2-D by rank or shaped tensor quickly (uniform + one decimal)."""
#     if isinstance(shape_or_rank, int):
#         t = rand_tensor((shape_or_rank,), dtype).to(device)
#     else:
#         t = rand_tensor(tuple(shape_or_rank), dtype).to(device)
#     return t


# @pytest.mark.usefixtures("ie_device", "precision")
# def test_multi_dot_errors_like_pytorch_cpu():
#     """
#     Mirrors PyTorch error checks: ranks, dtype/device mismatches, shape mismatch.
#     These validate *API semantics* only, so they call torch.linalg.multi_dot directly.
#     """
#     device = "cpu"
#     dtype = torch.float32
#     a = _mk(device, dtype, 2)

#     # At least two tensors
#     with pytest.raises(RuntimeError, match="expected at least 2 tensors"):
#         torch.linalg.multi_dot([])
#     with pytest.raises(RuntimeError, match="expected at least 2 tensors"):
#         torch.linalg.multi_dot([a])

#     # First/last tensor must be 1D or 2D
#     with pytest.raises(RuntimeError, match="the first tensor must be 1D or 2D"):
#         torch.linalg.multi_dot([torch.tensor(1, device=device, dtype=dtype), a])
#     with pytest.raises(RuntimeError, match="the last tensor must be 1D or 2D"):
#         torch.linalg.multi_dot([a, torch.tensor(1, device=device, dtype=dtype)])

#     # Middle tensors must be 2D
#     with pytest.raises(RuntimeError, match="tensor 1 must be 2D"):
#         torch.linalg.multi_dot([a, _mk(device, dtype, (2, 2, 2)), a])

#     # Dtype mismatch between inputs
#     with pytest.raises(RuntimeError, match="all tensors must have be the same dtype"):
#         torch.linalg.multi_dot([a, _mk(device, torch.double, 2)])

#     # Shape mismatch (cannot be multiplied)
#     with pytest.raises(RuntimeError, match="cannot be multiplied"):
#         torch.linalg.multi_dot([a, _mk(device, dtype, 3)])
#     with pytest.raises(RuntimeError, match="cannot be multiplied"):
#         torch.linalg.multi_dot([a, _mk(device, dtype, (3, 2)), a])


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.usefixtures("ie_device", "precision")
# def test_multi_dot_errors_like_pytorch_cuda():
#     """
#     Same API error checks but for CUDA device cases:
#     - all tensors must be on the same device
#     - 'out' must be on the same device & dtype
#     """
#     device = "cuda"
#     dtype = torch.float32
#     a = _mk(device, dtype, 2)

#     # Different device in inputs
#     with pytest.raises(RuntimeError, match="all tensors must be on the same device"):
#         torch.linalg.multi_dot([a, _mk("cpu", dtype, 2)])

#     # 'out' on different device / dtype
#     with pytest.raises(RuntimeError, match="expected out tensor to be on device"):
#         torch.linalg.multi_dot([a, a], out=torch.empty(0, dtype=dtype, device="cpu"))
#     with pytest.raises(RuntimeError, match="expected out tensor to have dtype"):
#         torch.linalg.multi_dot([a, a], out=torch.empty(0, dtype=torch.double, device=device))


# # ---------- Structure test: verify optimal parenthesization (MCO) ------------

# def _matmul_cost(shape_a, shape_b):
#     """
#     Scalar multiply count for one MatMul with 2D inputs:
#       (m x k) @ (k x n)  ->  m * k * n
#     """
#     assert len(shape_a) == 2 and len(shape_b) == 2, "expect 2D shapes"
#     m, k1 = shape_a
#     k2, n = shape_b
#     assert k1 == k2, f"incompatible shapes {shape_a} and {shape_b}"
#     return int(m) * int(k1) * int(n)


# def _dp_optimal_cost(dims):
#     """
#     Classic Matrix Chain Order DP.
#     dims is a list like [p0, p1, ..., pn] for matrices:
#       A0: p0 x p1, A1: p1 x p2, ..., An-1: p(n-1) x pn
#     Returns the minimal scalar multiply count.
#     """
#     n = len(dims) - 1
#     m = [[0]*(n) for _ in range(n)]
#     for L in range(2, n+1):
#         for i in range(0, n-L+1):
#             j = i + L - 1
#             best = None
#             for k in range(i, j):
#                 cost = m[i][k] + m[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
#                 best = cost if best is None else min(best, cost)
#             m[i][j] = best
#     return m[0][n-1] if n >= 1 else 0


# @pytest.mark.parametrize("shapes", [
#     # 3-matrix chain (translator should use PyTorch’s heuristic)
#     ([(40, 10), (10, 30), (30, 5)]),
#     # 4-matrix chain (translator should run full DP/MCO when static 2D)
#     ([(4, 8), (8, 16), (16, 5), (5, 7)]),
#     ([(10, 20), (20, 30), (30, 5), (5, 9)]),
# ])
# def test_multi_dot_structure_is_optimal_mco(shapes, ie_device, precision):
#     """
#     Convert a model that calls torch.linalg.multi_dot([...]).
#     Inspect OV graph and verify that the realized MatMul cost equals
#     the minimal DP (MCO) cost for these static 2D shapes.
#     """
#     class M(torch.nn.Module):
#         def forward(self, *ts):
#             return torch.linalg.multi_dot(list(ts))

#     # Build example inputs (values don’t matter; only shapes are used).
#     dtype = torch.float32
#     tensors = [rand_tensor(s, dtype) for s in shapes]

#     # Convert through PyTorch Frontend so it exercises aten::linalg_multi_dot
#     ov_model = ov.convert_model(M().eval(), example_input=tuple(tensors))

#     # Collect all MatMul nodes and sum their costs using their input shapes.
#     # Note: shapes are static for these cases.
#     cost_realized = 0
#     matmul_nodes = [n for n in ov_model.get_ops() if n.get_type_name() == "MatMul"]
#     assert len(matmul_nodes) == len(shapes) - 1, "chain of N matrices must yield N-1 MatMuls"

#     for mm in matmul_nodes:
#         a_shape = list(mm.input_value(0).get_shape())
#         b_shape = list(mm.input_value(1).get_shape())
#         # insist on pure 2D here by construction
#         assert len(a_shape) == 2 and len(b_shape) == 2, f"unexpected ranks: {a_shape}, {b_shape}"
#         cost_realized += _matmul_cost(a_shape, b_shape)

#     # Build dims[] for DP from the original shapes
#     # A0: p0 x p1, A1: p1 x p2, ..., so dims = [A0[0], A0[1], A1[1], ..., A_{n-1}[1]]
#     dims = [shapes[0][0]] + [s[1] for s in shapes]
#     cost_opt = _dp_optimal_cost(dims)

#     assert cost_realized == cost_opt, f"realized cost {cost_realized} != optimal {cost_opt}"
# /home/tamar/myOptproject/openvino/tests/layer_tests/pytorch_tests/test_linalg_multi_dot.py

# tests/layer_tests/pytorch_tests/test_linalg_multi_dot.py
# import pytest
# import torch
# import numpy as np
# import openvino as ov
# from pathlib import Path

# # ---------- Random helpers: uniform [-10, 10] with one decimal ----------------

# def rand_tensor(shape, dtype):
#     """
#     Create a tensor uniformly in [-10, 10] with exactly one decimal place.
#     Works for empty/zero-sized dims as well.
#     """
#     t = torch.empty(shape, dtype=dtype).uniform_(-10.0, 10.0)
#     t = torch.round(t * 10) / 10  # one decimal
#     return t


# def rand_tensor_from_spec(spec, dtype):
#     """
#     spec can be:
#       * int  -> 1-D tensor with that length
#       * tuple/list -> tensor with that shape
#     """
#     if isinstance(spec, (list, tuple)):
#         return rand_tensor(tuple(spec), dtype)
#     return rand_tensor((spec,), dtype)


# # ---------- Module that forces a CONSTANT list into linalg.multi_dot ----------

# class MConst(torch.nn.Module):
#     """
#     No-input module. Holds all matrices as constant buffers so that the list
#     passed to torch.linalg.multi_dot([...]) is CONST in the Torch graph.
#     This forces OV PyTorch Frontend to translate aten::linalg_multi_dot
#     (i.e., your translate_multi_dot).
#     """
#     def __init__(self, tensors):
#         super().__init__()
#         for i, t in enumerate(tensors):
#             self.register_buffer(f"T{i}", t.detach().clone())

#     def forward(self):
#         # buffers() preserves order of registration in PyTorch
#         ts = [getattr(self, f"T{i}") for i in range(len(list(self.buffers())))]
#         return torch.linalg.multi_dot(ts)


# # ---------- OV compile helpers & tolerances ----------------------------------

# def _compile(model, ie_device, precision, *, device_may_be_none=False):
#     """
#     Wrap ov.compile_model with useful defaults.
#     Force FP32 math on GPU when precision == 'FP32' to reduce numeric drift.
#     """
#     if device_may_be_none:
#         return ov.compile_model(model)  # AUTO
#     if ie_device == "GPU" and precision == "FP32":
#         return ov.compile_model(model, ie_device, {"INFERENCE_PRECISION_HINT": "f32"})
#     return ov.compile_model(model, ie_device)


# def _tolerances(ie_device, precision, *, chain_len=None, via_ir=False):
#     """
#     Choose rtol/atol per device/precision and scenario.

#     Base:
#       - CPU FP32: tight (1e-4, 1e-4)
#       - GPU FP32: slightly looser (1e-2, 1e-2)
#       - FP16 (any device): loose (1e-2, 1e-2)

#     Extra relaxations (targeted):
#       - FP16 + GPU + chain_len >= 3: a bit looser to cover realistic FP16 accumulation drift.
#       - FP16 + GPU + via_ir=True: a bit more loose due to IR round-trip numerics.
#     """
#     # Base tolerances
#     if precision == "FP16":
#         rtol, atol = 1e-2, 1e-2
#     elif ie_device == "GPU":
#         rtol, atol = 1e-2, 1e-2
#     else:
#         rtol, atol = 1e-4, 1e-4

#     # Long-ish chains in FP16 on GPU
#     if precision == "FP16" and ie_device == "GPU" and (chain_len or 0) >= 3:
#         rtol = max(rtol, 2.0e-2)
#         atol = max(atol, 5.0e-2)

#     # IR round-trip variance for FP16 + GPU
#     if precision == "FP16" and ie_device == "GPU" and via_ir:
#         rtol = max(rtol, 3.0e-2)
#         atol = max(atol, 8.0e-2)

#     return rtol, atol


# def _run_and_compare(tensors, ie_device, precision, *, via_ir=False, device_may_be_none=False):
#     """
#     Build a constant-list module calling torch.linalg.multi_dot([...]),
#     convert to OV (exercising aten::linalg_multi_dot translation),
#     compile, run, and compare numerically to PyTorch reference.
#     """
#     core = ov.Core()
#     if not device_may_be_none and ie_device not in core.available_devices:
#         pytest.skip(f"{ie_device} plugin not available (available: {core.available_devices})")

#     m = MConst(tensors).eval()

#     with torch.no_grad():
#         ref = torch.linalg.multi_dot(tensors)

#     model = ov.convert_model(m)  # no example_input: model has no parameters

#     if via_ir:
#         tmp = Path(".pytest_cache") / "ov_ir_tmp"
#         tmp.mkdir(parents=True, exist_ok=True)
#         xml_path = tmp / "model.xml"
#         bin_path = tmp / "model.bin"
#         ov.serialize(model, str(xml_path), str(bin_path))
#         model = core.read_model(str(xml_path))

#     compiled = _compile(model, ie_device, precision, device_may_be_none=device_may_be_none)

#     req = compiled.create_infer_request()
#     # No inputs to set; model has constant buffers only
#     req.infer()

#     out = req.get_output_tensor(0).data
#     assert out.shape == tuple(ref.shape)

#     rtol, atol = _tolerances(
#         ie_device,
#         precision,
#         chain_len=len(tensors),
#         via_ir=via_ir,
#     )
#     np.testing.assert_allclose(out, ref.detach().cpu().numpy(), rtol=rtol, atol=atol)


# # ---------- Functional tests (OV conversion + numeric check vs PyTorch) ------

# @pytest.mark.parametrize("shapes", [
#     # Basic 2D chain
#     ([(2, 3), (3, 4)]),                         # (2x3)(3x4) -> (2x4)
#     # 1D edges
#     ([(3,), (3, 4), (4,)]),                     # (3,) (3x4) (4,) -> scalar
#     # Longer 2D chain
#     ([(4, 8), (8, 16), (16, 5), (5, 7)]),       # -> (4x7)
# ])
# def test_linalg_multi_dot_convert_and_run(shapes, ie_device, precision):
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [rand_tensor(s, dtype) for s in shapes]
#     _run_and_compare(tensors, ie_device, precision)


# def test_multi_dot_two_vectors_scalar(ie_device, precision):
#     """1D · 1D -> scalar (0-D)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = rand_tensor(5, dtype)
#     b = rand_tensor(5, dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_vec_mat(ie_device, precision):
#     """(3,) × (3,4) -> (4,)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = rand_tensor(3, dtype)
#     b = rand_tensor((3, 4), dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_mat_vec(ie_device, precision):
#     """(3,4) × (4,) -> (3,)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = rand_tensor((3, 4), dtype)
#     b = rand_tensor(4, dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_three_with_vectors_scalar(ie_device, precision):
#     """(3,) × (3,4) × (4,) -> scalar (stresses 1D edges)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = rand_tensor(3, dtype)
#     b = rand_tensor((3, 4), dtype)
#     c = rand_tensor(4, dtype)
#     _run_and_compare([a, b, c], ie_device, precision)


# def test_multi_dot_inner_dim_one(ie_device, precision):
#     """Inner dimension = 1: (2,1) × (1,4) -> (2,4)."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a = rand_tensor((2, 1), dtype)
#     b = rand_tensor((1, 4), dtype)
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_noncontiguous_inputs(ie_device, precision):
#     """Non-contiguous storage (slice/transpose) should behave identically."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     a_full = rand_tensor((6, 4), dtype)
#     a = a_full[::2]               # (3,4) non-contiguous
#     b_full = rand_tensor((8, 4), dtype)
#     b = b_full[:4].t()            # (4,4) transposed -> non-contiguous
#     _run_and_compare([a, b], ie_device, precision)


# def test_multi_dot_via_ir_roundtrip(ie_device, precision):
#     """Serialize -> read_model IR round-trip; graph must survive."""
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [
#         rand_tensor((10, 100), dtype),
#         rand_tensor((100, 5),  dtype),
#         rand_tensor((5, 50),   dtype),
#     ]
#     _run_and_compare(tensors, ie_device, precision, via_ir=True)


# def test_multi_dot_auto_device_selection(ie_device, precision):
#     """Compile without specifying device (AUTO) if there is any available device."""
#     core = ov.Core()
#     if not core.available_devices:
#         pytest.skip("No devices available for AUTO run")

#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [
#         rand_tensor((2, 3), dtype),
#         rand_tensor((3, 4), dtype),
#     ]
#     _run_and_compare(tensors, ie_device, precision, device_may_be_none=True)


# # ---------- Extra functional coverage (parity-like) ---------------------------

# @pytest.mark.parametrize("shapes", [
#     # Variable output ranks
#     ([(2,), (2,)]),          # -> scalar
#     ([(1, 2), (2,)]),        # -> (1,)
#     ([(2,), (2, 1)]),        # -> (1,)
#     ([(1, 2), (2, 1)]),      # -> (1,1) -> scalar
#     ([(3, 2), (2, 4)]),      # -> (3,4)
#     # Multi-input chains
#     ([(3,), (3, 4), (4, 2), (2, 5), (5,)]),  # -> scalar
#     ([(1, 2), (2, 2), (2, 3), (3, 1)]),      # -> (1,1) -> scalar
#     # Larger cases (still moderate sizes)
#     ([(10, 20), (20, 30), (30, 5)]),
# ])
# def test_multi_dot_extra_functional(shapes, ie_device, precision):
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [rand_tensor(s, dtype) for s in shapes]
#     _run_and_compare(tensors, ie_device, precision)


# @pytest.mark.parametrize("shapes", [
#     # Empty-dimension paths (PyTorch allows these; OV matmul path usually handles them)
#     ([0], [0]),
#     ([2], [2, 0]),
#     ([1, 0], [0]),
#     ([0, 2], [2, 1]),
#     ([2, 2], [2, 0]),
#     ([2, 0], [0, 3]),
#     ([0, 0], [0, 1]),
#     ([4, 2], [2, 0], [0, 3], [3, 2]),
# ])
# def test_multi_dot_empty_dims(shapes, ie_device, precision):
#     torch.manual_seed(0)
#     dtype = torch.float16 if precision == "FP16" else torch.float32
#     tensors = [rand_tensor_from_spec(s, dtype) for s in shapes]
#     _run_and_compare(tensors, ie_device, precision)


# # ---------- API error parity (PyTorch semantics only; no OV conversion) ------

# def _mk(device, dtype, shape_or_rank):
#     """Quick helper to make either 1-D/2-D by rank or shaped tensor (uniform + one decimal)."""
#     if isinstance(shape_or_rank, int):
#         t = rand_tensor((shape_or_rank,), dtype).to(device)
#     else:
#         t = rand_tensor(tuple(shape_or_rank), dtype).to(device)
#     return t


# @pytest.mark.usefixtures("ie_device", "precision")
# def test_multi_dot_errors_like_pytorch_cpu():
#     device = "cpu"
#     dtype = torch.float32
#     a = _mk(device, dtype, 2)

#     # At least two tensors
#     with pytest.raises(RuntimeError, match="expected at least 2 tensors"):
#         torch.linalg.multi_dot([])
#     with pytest.raises(RuntimeError, match="expected at least 2 tensors"):
#         torch.linalg.multi_dot([a])

#     # First/last tensor must be 1D or 2D
#     with pytest.raises(RuntimeError, match="the first tensor must be 1D or 2D"):
#         torch.linalg.multi_dot([torch.tensor(1, device=device, dtype=dtype), a])
#     with pytest.raises(RuntimeError, match="the last tensor must be 1D or 2D"):
#         torch.linalg.multi_dot([a, torch.tensor(1, device=device, dtype=dtype)])

#     # Middle tensors must be 2D
#     with pytest.raises(RuntimeError, match="tensor 1 must be 2D"):
#         torch.linalg.multi_dot([a, _mk(device, dtype, (2, 2, 2)), a])

#     # Dtype mismatch between inputs
#     with pytest.raises(RuntimeError, match="all tensors must have be the same dtype"):
#         torch.linalg.multi_dot([a, _mk(device, torch.double, 2)])

#     # Shape mismatch (cannot be multiplied)
#     with pytest.raises(RuntimeError, match="cannot be multiplied"):
#         torch.linalg.multi_dot([a, _mk(device, dtype, 3)])
#     with pytest.raises(RuntimeError, match="cannot be multiplied"):
#         torch.linalg.multi_dot([a, _mk(device, dtype, (3, 2)), a])


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.usefixtures("ie_device", "precision")
# def test_multi_dot_errors_like_pytorch_cuda():
#     device = "cuda"
#     dtype = torch.float32
#     a = _mk(device, dtype, 2)

#     # Different device in inputs
#     with pytest.raises(RuntimeError, match="all tensors must be on the same device"):
#         torch.linalg.multi_dot([a, _mk("cpu", dtype, 2)])

#     # 'out' on different device / dtype
#     with pytest.raises(RuntimeError, match="expected out tensor to be on device"):
#         torch.linalg.multi_dot([a, a], out=torch.empty(0, dtype=dtype, device="cpu"))
#     with pytest.raises(RuntimeError, match="expected out tensor to have dtype"):
#         torch.linalg.multi_dot([a, a], out=torch.empty(0, dtype=torch.double, device=device))


# # ---------- Structure test: verify optimal parenthesization (MCO) ------------

# def _matmul_cost(shape_a, shape_b):
#     """
#     Scalar multiply count for one MatMul with 2D inputs:
#       (m x k) @ (k x n)  ->  m * k * n
#     """
#     assert len(shape_a) == 2 and len(shape_b) == 2, "expect 2D shapes"
#     m, k1 = shape_a
#     k2, n = shape_b
#     assert k1 == k2, f"incompatible shapes {shape_a} and {shape_b}"
#     return int(m) * int(k1) * int(n)


# def _dp_optimal_cost(dims):
#     """
#     Classic Matrix Chain Order DP.
#     dims is a list like [p0, p1, ..., pn] for matrices:
#       A0: p0 x p1, A1: p1 x p2, ..., An-1: p(n-1) x pn
#     Returns the minimal scalar multiply count.
#     """
#     n = len(dims) - 1
#     m = [[0]*(n) for _ in range(n)]
#     for L in range(2, n+1):
#         for i in range(0, n-L+1):
#             j = i + L - 1
#             best = None
#             for k in range(i, j):
#                 cost = m[i][k] + m[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
#                 best = cost if best is None else min(best, cost)
#             m[i][j] = best
#     return m[0][n-1] if n >= 1 else 0


# @pytest.mark.parametrize("shapes", [
#     # 3-matrix chain (translator should use PyTorch’s heuristic)
#     ([(40, 10), (10, 30), (30, 5)]),
#     # 4-matrix chain (translator should run full DP/MCO when static 2D)
#     ([(4, 8), (8, 16), (16, 5), (5, 7)]),
#     ([(10, 20), (20, 30), (30, 5), (5, 9)]),
# ])
# def test_multi_dot_structure_is_optimal_mco(shapes, ie_device, precision):
#     """
#     Convert a model that calls torch.linalg.multi_dot([...]) with a CONSTANT list.
#     Inspect OV graph and verify that the realized MatMul cost equals
#     the minimal DP (MCO) cost for these static 2D shapes.
#     """
#     # Build example inputs (values don’t matter; only shapes are used).
#     dtype = torch.float32
#     tensors = [rand_tensor(s, dtype) for s in shapes]

#     # Convert through PyTorch Frontend so it exercises aten::linalg_multi_dot
#     # ov_model = ov.convert_model(MConst(tensors).eval())
#     ov_model = ov.convert_model(MConst(tensors).eval(), example_input=())

#     # Collect all MatMul nodes and sum their costs using their input shapes.
#     cost_realized = 0
#     matmul_nodes = [n for n in ov_model.get_ops() if n.get_type_name() == "MatMul"]
#     assert len(matmul_nodes) == len(shapes) - 1, "chain of N matrices must yield N-1 MatMuls"

#     for mm in matmul_nodes:
#         a_shape = list(mm.input_value(0).get_shape())
#         b_shape = list(mm.input_value(1).get_shape())
#         assert len(a_shape) == 2 and len(b_shape) == 2, f"unexpected ranks: {a_shape}, {b_shape}"
#         cost_realized += _matmul_cost(a_shape, b_shape)

#     # Build dims[] for DP from the original shapes
#     dims = [shapes[0][0]] + [s[1] for s in shapes]
#     cost_opt = _dp_optimal_cost(dims)

#     assert cost_realized == cost_opt, f"realized cost {cost_realized} != optimal {cost_opt}"
# -*- coding: utf-8 -*-
# 🇬🇧 Tests for translate_multi_dot: numerical correctness + efficient multiplication order.
# 🇮🇱 טסטים ל-translate_multi_dot: נכונות חישובית + בדיקת סדר כפל יעיל.

# -*- coding: utf-8 -*-
# 🇬🇧 Tests for translate_multi_dot: numerical correctness + order-efficiency.
# 🇮🇱 טסטים ל-translate_multi_dot: נכונות חישובית + בדיקת סדר כפל יעיל.

# -*- coding: utf-8 -*-
# 🇬🇧 Tests for translate_multi_dot: numerical correctness + order-efficiency.
# 🇮🇱 טסטים ל-translate_multi_dot: נכונות חישובית + בדיקת סדר כפל יעיל.

import numpy as np
import pytest
import torch
import openvino as ov

RTOL = 1e-4
ATOL = 1e-4

# ---------- Helpers ----------

def make_small_tensor(shape, dtype=torch.float32, device="cpu", seed=0):
    """
    🇬🇧 Create a tensor with values in [-10, 10] and at most 1 decimal.
    🇮🇱 יוצר טנזור עם ערכים בתחום [-10, 10] ובעל ספרה עשרונית אחת לכל היותר.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    data = (torch.randint(-100, 101, shape, generator=g, device=device, dtype=torch.int32).to(torch.float32)) / 10.0
    return data.to(dtype)

def run_pt_and_ov(tensors, ie_device, precision, dtype=torch.float32):
    """
    🇬🇧 Run PyTorch multi_dot and OpenVINO-converted model on the same inputs.
    🇮🇱 מריץ גם את PyTorch וגם את OpenVINO על אותם קלטים.
    """
   
    class M(torch.nn.Module):
        def forward(self, *xs):
            return torch.linalg.multi_dot(xs)
    model = M().eval()
    example = tuple(tensors)

    ov_model = ov.convert_model(model, example_input=example)
    compiled = ov.Core().compile_model(
        ov_model,
        ie_device,
        config={"INFERENCE_PRECISION_HINT": str(precision)}  # e.g., "FP32" / "FP16"
    )

    with torch.no_grad():
        pt_out = model(*example).cpu().numpy()

    ov_inputs = [x.cpu().numpy() for x in example]
    ov_res = compiled(ov_inputs)[0]
    return pt_out, ov_res, ov_model

def param_sources(node, cache):
    """
    🇬🇧 Recursively collect indices of original Parameters that feed into 'node'.
    🇮🇱 מחזיר את סט האינדקסים של פרמטרים המקוריים שמזינים את 'node'.
    """
    if node in cache:
        return cache[node]
    sources = set()
    if node.get_type_name() == "Parameter":
        name = node.get_friendly_name()
        try:
            idx = int(name.rsplit("_", 1)[-1])
        except Exception:
            idx = None
        if idx is not None:
            sources.add(idx)
    else:
        for inp in node.inputs():
            src_node = inp.get_source_output().get_node()
            sources |= param_sources(src_node, cache)
    cache[node] = sources
    return sources

def first_level_matmuls(ov_model):
    """
    🇬🇧 Find MatMul nodes and map each to the set of source Parameter indices on its inputs.
    🇮🇱 מאתר צמדי MatMul ומחזיר עבור כל אחד את קבוצות הפרמטרים שמזינות אותו.
    """
    order = ov_model.get_ordered_ops()
    cache = {}
    mm = []
    for n in order:
        if n.get_type_name() == "MatMul":
            left = n.input_value(0).get_node()
            right = n.input_value(1).get_node()
            ls = param_sources(left, cache)
            rs = param_sources(right, cache)
            mm.append((n, frozenset(ls), frozenset(rs)))
    return mm

def assert_close(a, b):
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

# ---------- Tests ----------
# NOTE: All tests accept 'ie_device' and 'precision' per OpenVINO layer_tests harness.

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_multi_dot_two_tensors(ie_device, precision, dtype):
    # 🇮🇱 שני טנזורים – MatMul בודד.
    A = make_small_tensor((3, 4), dtype=dtype, seed=1)
    B = make_small_tensor((4, 2), dtype=dtype, seed=2)
    pt, ovv, _ = run_pt_and_ov([A, B], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_1d_endpoints(ie_device, precision, dtype):
    # 🇮🇱 קצוות 1D – Unsqueeze ל-(1,N)/(N,1) ואז Squeeze חזרה ל-1D.
    a = make_small_tensor((5,), dtype=dtype, seed=3)      # (5,)
    B = make_small_tensor((5, 7), dtype=dtype, seed=4)    # (5,7)
    c = make_small_tensor((7,), dtype=dtype, seed=5)      # (7,)
    pt, ovv, _ = run_pt_and_ov([a, B, c], ie_device, precision, dtype=dtype)
    assert (pt.ndim in (0, 1))
    assert_close(pt, ovv)

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_three_tensors_heuristic_picks_AB_then_C(ie_device, precision, dtype):
    # 🇮🇱 שלוש מטריצות – מקרה שבו זול יותר לחשב (A@B)@C מאשר A@(B@C).
    # A: (8x50), B: (50x6), C: (6x40)
    A = make_small_tensor((8, 50), dtype=dtype, seed=10)
    B = make_small_tensor((50, 6), dtype=dtype, seed=11)
    C = make_small_tensor((6, 40), dtype=dtype, seed=12)
    pt, ovv, ov_model = run_pt_and_ov([A, B, C], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)

    # 🇮🇱 בדיקת סדר: מצפים ל-MatMul ביניים שמחבר {0,1} לפני שילוב עם {2}.
    mms = first_level_matmuls(ov_model)
    has_ab = any(({0}.issubset(L) and {1}.issubset(R)) or ({1}.issubset(L) and {0}.issubset(R))
                 for _, L, R in mms)
    assert has_ab, "Expected heuristic to compute (A@B) first"

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_three_tensors_heuristic_picks_A_then_BC(ie_device, precision, dtype):
    # 🇮🇱 שלוש מטריצות – מקרה שבו זול יותר A@(B@C) מאשר (A@B)@C.
    # A: (40x6), B: (6x50), C: (50x8)
    A = make_small_tensor((40, 6), dtype=dtype, seed=20)
    B = make_small_tensor((6, 50), dtype=dtype, seed=21)
    C = make_small_tensor((50, 8), dtype=dtype, seed=22)
    pt, ovv, ov_model = run_pt_and_ov([A, B, C], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)

    mms = first_level_matmuls(ov_model)
    has_bc = any(({1}.issubset(L) and {2}.issubset(R)) or ({2}.issubset(L) and {1}.issubset(R))
                 for _, L, R in mms)
    assert has_bc, "Expected heuristic to compute (B@C) first"

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_four_tensors_dp_optimal_split(ie_device, precision, dtype):
    # 🇮🇱 ארבע מטריצות – בדיקת DP עם פיצול אופטימלי ((A@B)@(C@D)).
    A = make_small_tensor((5, 60), dtype=dtype, seed=30)
    B = make_small_tensor((60, 3), dtype=dtype, seed=31)
    C = make_small_tensor((3, 70), dtype=dtype, seed=32)
    D = make_small_tensor((70, 4), dtype=dtype, seed=33)
    pt, ovv, ov_model = run_pt_and_ov([A, B, C, D], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)

    mms = first_level_matmuls(ov_model)
    have_ab = any((L == frozenset({0}) and R == frozenset({1})) or
                  (L == frozenset({1}) and R == frozenset({0})) for _, L, R in mms)
    have_cd = any((L == frozenset({2}) and R == frozenset({3})) or
                  (L == frozenset({3}) and R == frozenset({2})) for _, L, R in mms)
    assert have_ab and have_cd, "Expected DP to form (A@B) and (C@D) as sub-products"

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_variable_shapes_small(ie_device, precision, dtype):
    # 🇮🇱 מקרים קטנים/רנדומליים (כולל 1D בקצה אחד), כדי לכסות מסלולי קוד שונים.
    a = make_small_tensor((4,), dtype=dtype, seed=40)      # (4,)
    B = make_small_tensor((4, 3), dtype=dtype, seed=41)    # (4,3)
    C = make_small_tensor((3, 2), dtype=dtype, seed=42)    # (3,2)
    d = make_small_tensor((2,), dtype=dtype, seed=43)      # (2,)
    pt, ovv, _ = run_pt_and_ov([a, B, C, d], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)
import numpy as np
import torch
import openvino as ov

RTOL_MDOT = 1e-4
ATOL_MDOT = 1e-4


def make_small_tensor(shape, dtype=torch.float32, device="cpu", seed=0):
    """Create a small deterministic tensor in [-10, 10] with one decimal digit."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    data = (
        torch.randint(-100, 101, shape, generator=g, device=device, dtype=torch.int32)
        .to(torch.float32)
        / 10.0
    )
    return data.to(dtype)


def run_pt_and_ov_multi_dot(tensors, ie_device, precision, dtype=torch.float32):
    """Run PyTorch torch.linalg.multi_dot and the OpenVINO-converted model on the same inputs."""

    class MultiDotWrapper(torch.nn.Module):
        def forward(self, *xs):
            return torch.linalg.multi_dot(xs)

    model = MultiDotWrapper().eval()
    example = tuple(tensors)

    # Use the robust wrapper that works with/without openvino.convert_model
    ov_model = ov_convert(model, example_input=example)

    core = ov.Core()
    compiled = core.compile_model(
        ov_model,
        ie_device,
        config={"INFERENCE_PRECISION_HINT": str(precision)},
    )

    with torch.no_grad():
        pt_out = model(*example).detach().cpu().numpy()

    ov_inputs = [x.detach().cpu().numpy() for x in example]
    ov_res_map = compiled.create_infer_request().infer(ov_inputs)
    # Single-output model -> take first value
    ov_out = next(iter(ov_res_map.values()))

    return pt_out, ov_out, ov_model


def assert_close_multi_dot(a, b):
    """Helper for numerical comparison with multi_dot-specific tolerances."""
    np.testing.assert_allclose(a, b, rtol=RTOL_MDOT, atol=ATOL_MDOT)


def _param_sources(node, cache):
    """Recursively collect indices of original Parameters that feed into 'node'."""
    if node in cache:
        return cache[node]

    sources = set()
    if node.get_type_name() == "Parameter":
        name = node.get_friendly_name()
        try:
            idx = int(name.rsplit("_", 1)[-1])
        except Exception:
            idx = None
        if idx is not None:
            sources.add(idx)
    else:
        for inp in node.inputs():
            src_node = inp.get_source_output().get_node()
            sources |= _param_sources(src_node, cache)

    cache[node] = sources
    return sources


def collect_matmul_param_sets(ov_model):
    """
    Collect all MatMul nodes and, for each one, the sets of source Parameter indices
    on its left and right inputs.

    Returns:
        List of tuples: (matmul_node, left_sources, right_sources)
        where left_sources/right_sources are frozenset[int] of parameter indices.
    """
    order = ov_model.get_ordered_ops()
    cache = {}
    mm = []
    for n in order:
        if n.get_type_name() == "MatMul":
            left_node = n.input_value(0).get_node()
            right_node = n.input_value(1).get_node()
            ls = _param_sources(left_node, cache)
            rs = _param_sources(right_node, cache)
            mm.append((n, frozenset(ls), frozenset(rs)))
    return mm
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_multi_dot_three_tensors_prefers_AB_then_C(ie_device, precision, ir_version, dtype):
    """
    Three matrices case where (A @ B) @ C is cheaper than A @ (B @ C).
    We expect the translator's heuristic to form the AB product first.
    """
    # Shapes:
    #   A: (8, 50)
    #   B: (50, 6)
    #   C: (6, 40)
    # Cost((A @ B) @ C) = 8*50*6 + 8*6*40 = 2400 + 1920 = 4320
    # Cost(A @ (B @ C)) = 50*6*40 + 8*50*40 = 12000 + 16000 = 28000
    A = make_small_tensor((8, 50), dtype=dtype, seed=100)
    B = make_small_tensor((50, 6), dtype=dtype, seed=101)
    C = make_small_tensor((6, 40), dtype=dtype, seed=102)

    pt_out, ov_out, ov_model = run_pt_and_ov_multi_dot(
        [A, B, C],
        ie_device=ie_device,
        precision=precision,
        dtype=dtype,
    )
    assert_close_multi_dot(pt_out, ov_out)

    # Check structure: there must exist a MatMul whose inputs come exactly from {0,1}
    # (i.e. it forms AB as a sub-product before combining with C).
    matmuls = collect_matmul_param_sets(ov_model)
    has_ab = any(
        (L == frozenset({0}) and R == frozenset({1}))
        or (L == frozenset({1}) and R == frozenset({0}))
        for _, L, R in matmuls
    )
    assert has_ab, "Expected heuristic to compute (A @ B) as an intermediate sub-product"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_multi_dot_three_tensors_prefers_A_then_BC(ie_device, precision, ir_version, dtype):
    """
    Three matrices case where A @ (B @ C) is cheaper than (A @ B) @ C.
    We expect the translator's heuristic to form the BC product first.
    """
    # Shapes:
    #   A: (40, 6)
    #   B: (6, 50)
    #   C: (50, 8)
    # Cost((A @ B) @ C) = 40*6*50 + 40*50*8 = 12000 + 16000 = 28000
    # Cost(A @ (B @ C)) = 6*50*8 + 40*6*8 = 2400 + 1920 = 4320
    A = make_small_tensor((40, 6), dtype=dtype, seed=200)
    B = make_small_tensor((6, 50), dtype=dtype, seed=201)
    C = make_small_tensor((50, 8), dtype=dtype, seed=202)

    pt_out, ov_out, ov_model = run_pt_and_ov_multi_dot(
        [A, B, C],
        ie_device=ie_device,
        precision=precision,
        dtype=dtype,
    )
    assert_close_multi_dot(pt_out, ov_out)

    # Check structure: there must exist a MatMul whose inputs come exactly from {1,2}
    # (i.e. it forms BC as a sub-product before combining with A).
    matmuls = collect_matmul_param_sets(ov_model)
    has_bc = any(
        (L == frozenset({1}) and R == frozenset({2}))
        or (L == frozenset({2}) and R == frozenset({1}))
        for _, L, R in matmuls
    )
    assert has_bc, "Expected heuristic to compute (B @ C) as an intermediate sub-product"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_multi_dot_four_tensors_dp_forms_AB_and_CD(ie_device, precision, ir_version, dtype):
    """
    Four matrices case where the DP-style splitter should form (A @ B) and (C @ D)
    as independent sub-products before the final MatMul.
    """
    # Shapes chosen so that ((A @ B) @ (C @ D)) is naturally optimal for a DP planner.
    #   A: (5, 60)
    #   B: (60, 3)
    #   C: (3, 70)
    #   D: (70, 4)
    A = make_small_tensor((5, 60), dtype=dtype, seed=300)
    B = make_small_tensor((60, 3), dtype=dtype, seed=301)
    C = make_small_tensor((3, 70), dtype=dtype, seed=302)
    D = make_small_tensor((70, 4), dtype=dtype, seed=303)

    pt_out, ov_out, ov_model = run_pt_and_ov_multi_dot(
        [A, B, C, D],
        ie_device=ie_device,
        precision=precision,
        dtype=dtype,
    )
    assert_close_multi_dot(pt_out, ov_out)

    matmuls = collect_matmul_param_sets(ov_model)

    # Look for explicit (A @ B): parameters {0} and {1}
    have_ab = any(
        (L == frozenset({0}) and R == frozenset({1}))
        or (L == frozenset({1}) and R == frozenset({0}))
        for _, L, R in matmuls
    )

    # Look for explicit (C @ D): parameters {2} and {3}
    have_cd = any(
        (L == frozenset({2}) and R == frozenset({3}))
        or (L == frozenset({3}) and R == frozenset({2}))
        for _, L, R in matmuls
    )

    assert have_ab and have_cd, "Expected DP to form (A @ B) and (C @ D) as sub-products"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_multi_dot_matmul_count_is_n_minus_1(ie_device, precision, ir_version, dtype):
    """
    Sanity check: for N tensors, the resulting graph should contain at least N-1 MatMul nodes.
    This ensures no degenerate lowering that would expand into something more expensive.
    """
    # Example with 5 tensors in a valid chain:
    A = make_small_tensor((4, 10), dtype=dtype, seed=400)
    B = make_small_tensor((10, 6), dtype=dtype, seed=401)
    C = make_small_tensor((6, 3), dtype=dtype, seed=402)
    D = make_small_tensor((3, 8), dtype=dtype, seed=403)
    E = make_small_tensor((8, 2), dtype=dtype, seed=404)

    tensors = [A, B, C, D, E]
    pt_out, ov_out, ov_model = run_pt_and_ov_multi_dot(
        tensors,
        ie_device=ie_device,
        precision=precision,
        dtype=dtype,
    )
    assert_close_multi_dot(pt_out, ov_out)

    matmuls = [n for n in ov_model.get_ordered_ops() if n.get_type_name() == "MatMul"]
    assert len(matmuls) >= len(tensors) - 1, (
        f"Expected at least {len(tensors) - 1} MatMul nodes for "
        f"{len(tensors)}-tensor multi_dot chain, got {len(matmuls)}"
    )
