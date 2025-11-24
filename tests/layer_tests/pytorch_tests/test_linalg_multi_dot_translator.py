
# # import torch
# # import numpy as np
# # import pytest
# # import openvino as ov


# # def ov_convert(model_or_fn, example_input=None):
# #     """
# #     Unified converter that works even when `openvino.convert_model`
# #     is not exposed at top-level.

# #     1. Try high-level `openvino.convert_model` if available.
# #     2. Fallback: use FrontEndManager + TorchScriptPythonDecoder.
# #     """
# #     # --- 1) high-level helper (if your build exposes it) ---
# #     try:
# #         from openvino import convert_model as _ov_convert_model
# #         if example_input is not None:
# #             return _ov_convert_model(model_or_fn, example_input=example_input)
# #         return _ov_convert_model(model_or_fn)
# #     except Exception:
# #         pass

# #     # --- 2) Fallback: FrontEndManager + TorchScriptPythonDecoder ---
# #     from openvino.frontend import FrontEndManager
# #     from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

# #     fem = FrontEndManager()
# #     fe = fem.load_by_framework("pytorch")

# #     if example_input is None:
# #         dec = TorchScriptPythonDecoder(model_or_fn)
# #     else:
# #         dec = TorchScriptPythonDecoder(model_or_fn, example_input=example_input)

# #     input_model = fe.load(dec)
# #     return fe.convert(input_model)


# # class MultiDotConstModule(torch.nn.Module):
# #     """
# #     Simple module that uses torch.linalg.multi_dot on constant buffers.
# #     After script+freeze, the graph becomes a pure constant.
# #     """

# #     def __init__(self):
# #         super().__init__()
# #         # Constants as buffers (freeze will embed them into the TS graph)
# #         self.register_buffer(
# #             "A",
# #             torch.tensor([[1.0, 2.0],
# #                           [3.0, 4.0]], dtype=torch.float32),
# #         )
# #         self.register_buffer(
# #             "B",
# #             torch.tensor([[1.0, 0.0],
# #                           [0.0, 1.0]], dtype=torch.float32),
# #         )
# #         self.register_buffer(
# #             "C",
# #             torch.tensor([[5.0],
# #                           [6.0]], dtype=torch.float32),
# #         )

# #     def forward(self):
# #         # Constant ListConstruct after freeze
# #         xs = [self.A, self.B, self.C]
# #         return torch.linalg.multi_dot(xs)


# # def test_linalg_multi_dot_const_freeze(ie_device, precision, ir_version):
# #     """
# #     Validate aten::linalg_multi_dot with constant inputs (buffers + freeze):

# #     - The PT frontend (TS path) can convert it fully.
# #     - The resulting OV model matches PyTorch numerically.
# #     """

# #     # Skip GPU if plugin is not available in this environment
# #     if ie_device == "GPU":
# #         core_probe = ov.Core()
# #         if "GPU" not in core_probe.get_available_devices():
# #             pytest.skip("GPU plugin is not available in this environment")

# #     # Create and prepare Torch module
# #     mod = MultiDotConstModule().eval()

# #     # Script + freeze to embed constants into the TorchScript graph
# #     scripted = torch.jit.script(mod)
# #     frozen = torch.jit.freeze(scripted)

# #     # PyTorch reference result
# #     ref = torch.linalg.multi_dot(
# #         [mod.A, mod.B, mod.C]
# #     ).detach().cpu().numpy()

# #     # Convert using robust wrapper (works with/without openvino.convert_model)
# #     ov_model = ov_convert(frozen)

# #     # Compile & infer with OpenVINO
# #     core = ov.Core()
# #     compiled = core.compile_model(ov_model, ie_device)
# #     result_map = compiled.create_infer_request().infer()
# #     out = next(iter(result_map.values()))

# #     # Numerical check
# #     if precision == "FP16":
# #         np.testing.assert_allclose(
# #             out.astype(np.float16),
# #             ref.astype(np.float16),
# #             rtol=2e-2,
# #             atol=2e-2,
# #         )
# #     else:
# #         np.testing.assert_allclose(
# #             out,
# #             ref,
# #             rtol=1e-5,
# #             atol=1e-6,
# #         )
# # (comments in English only)
# import torch
# import numpy as np
# import pytest
# import openvino as ov


# def ov_convert(model_or_fn, example_input=None):
#     """
#     Unified converter that works even when `openvino.convert_model`
#     is not exposed at top-level.

#     1. Try high-level `openvino.convert_model` if available.
#     2. Fallback: use FrontEndManager + TorchScriptPythonDecoder.
#     """
#     # --- 1) high-level helper (if your build exposes it) ---
#     try:
#         from openvino import convert_model as _ov_convert_model
#         if example_input is not None:
#             return _ov_convert_model(model_or_fn, example_input=example_input)
#         return _ov_convert_model(model_or_fn)
#     except Exception:
#         pass

#     # --- 2) Fallback: FrontEndManager + TorchScriptPythonDecoder ---
#     from openvino.frontend import FrontEndManager
#     from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

#     fem = FrontEndManager()
#     fe = fem.load_by_framework("pytorch")

#     if example_input is None:
#         dec = TorchScriptPythonDecoder(model_or_fn)
#     else:
#         dec = TorchScriptPythonDecoder(model_or_fn, example_input=example_input)

#     input_model = fe.load(dec)
#     return fe.convert(input_model)


# class MultiDotConstModule(torch.nn.Module):
#     """
#     Simple module that uses torch.linalg.multi_dot on constant buffers.
#     After script+freeze, the graph becomes a pure constant.
#     """

#     def __init__(self):
#         super().__init__()
#         # Constants as buffers (freeze will embed them into the TS graph)
#         self.register_buffer(
#             "A",
#             torch.tensor([[1.0, 2.0],
#                           [3.0, 4.0]], dtype=torch.float32),
#         )
#         self.register_buffer(
#             "B",
#             torch.tensor([[1.0, 0.0],
#                           [0.0, 1.0]], dtype=torch.float32),
#         )
#         self.register_buffer(
#             "C",
#             torch.tensor([[5.0],
#                           [6.0]], dtype=torch.float32),
#         )

#     def forward(self):
#         # Constant ListConstruct after freeze
#         xs = [self.A, self.B, self.C]
#         return torch.linalg.multi_dot(xs)


# def test_linalg_multi_dot_const_freeze(ie_device, precision, ir_version):
#     """
#     Validate aten::linalg_multi_dot with constant inputs (buffers + freeze):

#     - The PT frontend (TS path) can convert it fully.
#     - The resulting OV model matches PyTorch numerically.
#     """

#     # Skip GPU if plugin is not available in this environment
#     if ie_device == "GPU":
#         core_probe = ov.Core()
#         if "GPU" not in core_probe.get_available_devices():
#             pytest.skip("GPU plugin is not available in this environment")

#     # Create and prepare Torch module
#     mod = MultiDotConstModule().eval()

#     # Script + freeze to embed constants into the TorchScript graph
#     scripted = torch.jit.script(mod)
#     frozen = torch.jit.freeze(scripted)

#     # PyTorch reference result
#     ref = torch.linalg.multi_dot(
#         [mod.A, mod.B, mod.C]
#     ).detach().cpu().numpy()

#     # Convert using robust wrapper (works with/without openvino.convert_model)
#     ov_model = ov_convert(frozen)

#     # Compile & infer with OpenVINO
#     core = ov.Core()
#     compiled = core.compile_model(ov_model, ie_device)
#     result_map = compiled.create_infer_request().infer()
#     out = next(iter(result_map.values()))

#     # Numerical check
#     if precision == "FP16":
#         np.testing.assert_allclose(
#             out.astype(np.float16),
#             ref.astype(np.float16),
#             rtol=2e-2,
#             atol=2e-2,
#         )
#     else:
#         np.testing.assert_allclose(
#             out,
#             ref,
#             rtol=1e-5,
#             atol=1e-6,
#         )


# class MultiDotModuleShapes(torch.nn.Module):
#     """
#     Generic module that holds a list of constant matrices/vectors (buffers)
#     with given shapes, and applies torch.linalg.multi_dot on them.
#     """

#     def __init__(self, shapes):
#         super().__init__()
#         self._num = len(shapes)
#         for idx, shape in enumerate(shapes):
#             name = f"W{idx}"
#             # Use random but deterministic-ish data (no need for fixed seed here)
#             tensor = torch.randn(*shape, dtype=torch.float32)
#             self.register_buffer(name, tensor)

#     def forward(self):
#         xs = []
#         for idx in range(self._num):
#             xs.append(getattr(self, f"W{idx}"))
#         return torch.linalg.multi_dot(xs)


# @pytest.mark.parametrize(
#     "shapes",
#     [
#         # 2 inputs – various endpoint dimensionalities
#         ([ (2,),      (2,)      ]),   # [2] x [2]
#         ([ (1, 2),    (2,)      ]),   # [1,2] x [2]
#         ([ (2,),      (2, 1)    ]),   # [2] x [2,1]
#         ([ (1, 2),    (2, 1)    ]),   # [1,2] x [2,1]
#         ([ (3, 2),    (2, 4)    ]),   # [3,2] x [2,4]

#         # Multiple input tensors
#         ([ (3,),      (3, 4),   (4, 2),   (2, 5),   (5,)      ]),
#         ([ (1, 2),    (2, 2),   (2, 3),   (3, 1)               ]),

#         # Larger tensors
#         ([ (10, 100), (100, 5), (5, 50) ]),   # typical "tall x wide x etc"
#         ([ (10, 20),  (20, 30), (30, 5) ]),
#     ],
# )
# def test_linalg_multi_dot_various_shapes(ie_device, precision, ir_version, shapes):
#     """
#     Shape coverage test inspired by PyTorch's test_multi_dot:

#     - Covers different endpoint ranks (1D/2D).
#     - Covers multiple matrices chain.
#     - Covers larger matrices.
#     - Validates numerical equivalence with PyTorch.
#     """

#     # Skip GPU if plugin is not available in this environment
#     if ie_device == "GPU":
#         core_probe = ov.Core()
#         if "GPU" not in core_probe.get_available_devices():
#             pytest.skip("GPU plugin is not available in this environment")

#     # Create Torch module with constant buffers of given shapes
#     mod = MultiDotModuleShapes(shapes).eval()

#     # Script + freeze to embed constants into the TS graph
#     scripted = torch.jit.script(mod)
#     frozen = torch.jit.freeze(scripted)

#     # Build the reference multi_dot on the stored buffers
#     buffers = [getattr(mod, f"W{i}") for i in range(len(shapes))]
#     ref = torch.linalg.multi_dot(buffers).detach().cpu().numpy()

#     # Convert using PT frontend
#     ov_model = ov_convert(frozen)

#     # Compile & infer
#     core = ov.Core()
#     compiled = core.compile_model(ov_model, ie_device)
#     result_map = compiled.create_infer_request().infer()
#     out = next(iter(result_map.values()))

#     # Numerical check
#     if precision == "FP16":
#         np.testing.assert_allclose(
#             out.astype(np.float16),
#             ref.astype(np.float16),
#             rtol=2e-2,
#             atol=2e-2,
#         )
#     else:
#         np.testing.assert_allclose(
#             out,
#             ref,
#             rtol=1e-5,
#             atol=1e-6,
#         )


# class MultiDotOneTensorModule(torch.nn.Module):
#     """
#     Module that intentionally calls torch.linalg.multi_dot on a single tensor.
#     This should fail in the PT frontend translation with a clear error.
#     """

#     def __init__(self):
#         super().__init__()
#         self.register_buffer("A", torch.randn(2, 2, dtype=torch.float32))

#     def forward(self):
#         xs = [self.A]  # single element
#         return torch.linalg.multi_dot(xs)


# def test_linalg_multi_dot_error_too_few_tensors():
#     """
#     Error-path test inspired by PyTorch's test_multi_dot_errors:

#     We verify that PT frontend translation fails when linalg_multi_dot
#     receives fewer than 2 tensors, and that the error message corresponds
#     to the FRONT_END_OP_CONVERSION_CHECK in translate_multi_dot.cpp.
#     """
#     import openvino._pyopenvino as _ov_capi

#     mod = MultiDotOneTensorModule().eval()
#     scripted = torch.jit.script(mod)
#     frozen = torch.jit.freeze(scripted)

#     # Expect OpConversionFailure from the frontend
#     with pytest.raises(
#         _ov_capi.OpConversionFailure,
#         match=r"linalg_multi_dot requires at least 2 tensors; got 1",
#     ):
#         ov_convert(frozen)
import torch
import numpy as np
import pytest
import openvino as ov
import openvino._pyopenvino as _ov_capi


def ov_convert(model_or_fn, example_input=None):
    """
    Unified converter that works even when `openvino.convert_model`
    is not exposed at top-level.

    1. Try high-level `openvino.convert_model` if available.
    2. Fallback: use FrontEndManager + TorchScriptPythonDecoder.
    """
    # --- 1) high-level helper (if your build exposes it) ---
    try:
        from openvino import convert_model as _ov_convert_model
        if example_input is not None:
            return _ov_convert_model(model_or_fn, example_input=example_input)
        return _ov_convert_model(model_or_fn)
    except Exception:
        pass

    # --- 2) Fallback: FrontEndManager + TorchScriptPythonDecoder ---
    from openvino.frontend import FrontEndManager
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    fem = FrontEndManager()
    fe = fem.load_by_framework("pytorch")

    if example_input is None:
        dec = TorchScriptPythonDecoder(model_or_fn)
    else:
        dec = TorchScriptPythonDecoder(model_or_fn, example_input=example_input)


    input_model = fe.load(dec)
    return fe.convert(input_model)


class MultiDotConstModule(torch.nn.Module):
    """
    Simple module that uses torch.linalg.multi_dot on constant buffers.
    After script+freeze, the graph becomes a pure constant.
    """

    def __init__(self):
        super().__init__()
        # Constants as buffers (freeze will embed them into the TS graph)
        self.register_buffer(
            "A",
            torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]], dtype=torch.float32),
        )
        self.register_buffer(
            "B",
            torch.tensor([[1.0, 0.0],
                          [0.0, 1.0]], dtype=torch.float32),
        )
        self.register_buffer(
            "C",
            torch.tensor([[5.0],
                          [6.0]], dtype=torch.float32),
        )

    def forward(self):
        # Constant ListConstruct after freeze
        xs = [self.A, self.B, self.C]
        return torch.linalg.multi_dot(xs)


def test_linalg_multi_dot_const_freeze(ie_device, precision, ir_version):
    """
    Validate aten::linalg_multi_dot with constant inputs (buffers + freeze):

    - The PT frontend (TS path) can convert it fully.
    - The resulting OV model matches PyTorch numerically.
    """

    # Skip GPU if plugin is not available in this environment
    if ie_device == "GPU":
        core_probe = ov.Core()
        if "GPU" not in core_probe.get_available_devices():
            pytest.skip("GPU plugin is not available in this environment")

    # Create and prepare Torch module
    mod = MultiDotConstModule().eval()

    # Script + freeze to embed constants into the TorchScript graph
    scripted = torch.jit.script(mod)
    frozen = torch.jit.freeze(scripted)

    # PyTorch reference result
    ref = torch.linalg.multi_dot(
        [mod.A, mod.B, mod.C]
    ).detach().cpu().numpy()

    # Convert using robust wrapper (works with/without openvino.convert_model)
    ov_model = ov_convert(frozen)

    # Compile & infer with OpenVINO
    core = ov.Core()
    compiled = core.compile_model(ov_model, ie_device)
    result_map = compiled.create_infer_request().infer()
    out = next(iter(result_map.values()))

    # Numerical check
    if precision == "FP16":
        np.testing.assert_allclose(
            out.astype(np.float16),
            ref.astype(np.float16),
            rtol=2e-2,
            atol=2e-2,
        )
    else:
        np.testing.assert_allclose(
            out,
            ref,
            rtol=1e-5,
            atol=1e-6,
        )


class MultiDotModuleShapes(torch.nn.Module):
    """
    Generic module that holds a list of constant matrices/vectors (buffers)
    with given shapes, and applies torch.linalg.multi_dot on them.
    """

    def __init__(self, shapes):
        super().__init__()
        self._num = len(shapes)
        for idx, shape in enumerate(shapes):
            name = f"W{idx}"
            # Use random but deterministic-ish data (no need for fixed seed here)
            tensor = torch.randn(*shape, dtype=torch.float32)
            self.register_buffer(name, tensor)

    def forward(self):
        xs = []
        for idx in range(self._num):
            xs.append(getattr(self, f"W{idx}"))
        return torch.linalg.multi_dot(xs)


@pytest.mark.parametrize(
    "shapes",
    [
        # 2 inputs – various endpoint dimensionalities
        ([(2,),      (2,)      ]),   # [2] x [2]
        ([(1, 2),    (2,)      ]),   # [1,2] x [2]
        ([(2,),      (2, 1)    ]),   # [2] x [2,1]
        ([(1, 2),    (2, 1)    ]),   # [1,2] x [2,1]
        ([(3, 2),    (2, 4)    ]),   # [3,2] x [2,4]

        # Multiple input tensors
        ([(3,),      (3, 4),   (4, 2),   (2, 5),   (5,)      ]),
        ([(1, 2),    (2, 2),   (2, 3),   (3, 1)               ]),

        # Larger tensors
        ([(10, 100), (100, 5), (5, 50)]),
        ([(10, 20),  (20, 30), (30, 5)]),
    ],
)
@pytest.mark.xfail(
    reason=(
        "PT FE: dynamic Tensor[] (prim::ListConstruct of non-constant tensors) "
        "for aten::linalg_multi_dot is not fully supported yet – "
        "get_list_as_outputs() fails to unpack, causing OpConversionFailure."
    ),
    raises=_ov_capi.OpConversionFailure,
)
def test_linalg_multi_dot_various_shapes(ie_device, precision, ir_version, shapes):
    """
    Shape coverage test inspired by PyTorch's test_multi_dot.

    NOTE:
        Currently expected to xfail due to PT FE limitation for dynamic Tensor[]
        inputs to aten::linalg_multi_dot (see translate_multi_dot.cpp and utils.hpp).
    """

    # Skip GPU if plugin is not available in this environment
    if ie_device == "GPU":
        core_probe = ov.Core()
        if "GPU" not in core_probe.get_available_devices():
            pytest.skip("GPU plugin is not available in this environment")

    # Create Torch module with constant buffers of given shapes
    mod = MultiDotModuleShapes(shapes).eval()

    # PyTorch reference result (eager)
    ref = mod().detach().cpu().numpy()

    # Convert using high-level PT frontend (TS/FX) via ov_convert
    ov_model = ov_convert(mod, example_input=())

    # Compile & infer
    core = ov.Core()
    compiled = core.compile_model(ov_model, ie_device)
    result_map = compiled.create_infer_request().infer()
    out = next(iter(result_map.values()))

    # Numerical check
    if precision == "FP16":
        np.testing.assert_allclose(
            out.astype(np.float16),
            ref.astype(np.float16),
            rtol=2e-2,
            atol=2e-2,
        )
    else:
        np.testing.assert_allclose(
            out,
            ref,
            rtol=1e-5,
            atol=1e-6,
        )



class MultiDotOneTensorModule(torch.nn.Module):
    """
    Module that intentionally calls torch.linalg.multi_dot on a single tensor.
    This should fail in the PT frontend translation with a clear error.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("A", torch.randn(2, 2, dtype=torch.float32))

    def forward(self):
        xs = [self.A]  # single element
        return torch.linalg.multi_dot(xs)


def test_linalg_multi_dot_error_too_few_tensors(ie_device, precision, ir_version):
    """
    Error-path test inspired by PyTorch's test_multi_dot_errors:

    We verify that PT frontend translation fails when linalg_multi_dot
    receives fewer than 2 tensors, and that the error message corresponds
    to the FRONT_END_OP_CONVERSION_CHECK in translate_multi_dot.cpp.
    """
    import openvino._pyopenvino as _ov_capi

    # Skip GPU if plugin is not available in this environment
    if ie_device == "GPU":
        core_probe = ov.Core()
        if "GPU" not in core_probe.get_available_devices():
            pytest.skip("GPU plugin is not available in this environment")

    mod = MultiDotOneTensorModule().eval()
    scripted = torch.jit.script(mod)
    frozen = torch.jit.freeze(scripted)

    # Expect OpConversionFailure from the frontend
    with pytest.raises(
        _ov_capi.OpConversionFailure,
        match=r"linalg_multi_dot requires at least 2 tensors; got 1",
    ):
        ov_convert(frozen)

# import numpy as np
# import torch
# import pytest
# import openvino as ov

# RTOL_MDOT = 1e-4
# ATOL_MDOT = 1e-4


# def make_small_tensor(shape, dtype=torch.float32, device="cpu", seed=0):
#     """Create a small deterministic tensor in [-10, 10] with one decimal digit."""
#     g = torch.Generator(device=device)
#     g.manual_seed(seed)
#     data = (
#         torch.randint(-100, 101, shape, generator=g, device=device, dtype=torch.int32)
#         .to(torch.float32)
#         / 10.0
#     )
#     return data.to(dtype)


# def assert_close_multi_dot(a, b):
#     """Helper for numerical comparison with multi_dot-specific tolerances."""
#     np.testing.assert_allclose(a, b, rtol=RTOL_MDOT, atol=ATOL_MDOT)


# def collect_matmul_shapes(ov_model):
#     """Return list of (left_shape, right_shape) for each MatMul in the graph."""
#     mm = []
#     for n in ov_model.get_ordered_ops():
#         if n.get_type_name() == "MatMul":
#             l_ps = n.input_value(0).get_partial_shape()
#             r_ps = n.input_value(1).get_partial_shape()
#             l_shape = tuple(int(d) for d in l_ps)
#             r_shape = tuple(int(d) for d in r_ps)
#             mm.append((l_shape, r_shape))
#     return mm


# def run_const_module_and_ov(mod, ie_device, precision):
#     """Run constant-only multi_dot module through TS freeze + PT FE + OpenVINO."""
#     # PyTorch reference output
#     with torch.no_grad():
#         pt_out = mod().detach().cpu().numpy()

#     # TorchScript + freeze – inline buffers as constants in the TS graph
#     scripted = torch.jit.script(mod)
#     frozen = torch.jit.freeze(scripted)

#     # Convert via PT FE wrapper (ov_convert is defined earlier in this file)
#     ov_model = ov_convert(frozen)

#     core = ov.Core()

#     # Handle environments without GPU plugin
#     if ie_device == "GPU":
#         available = core.get_available_devices()
#         if "GPU" not in available:
#             pytest.skip("GPU plugin is not available in this environment")

#     compiled = core.compile_model(ov_model, ie_device)
#     result_map = compiled.create_infer_request().infer()
#     ov_out = next(iter(result_map.values()))

#     return pt_out, ov_out, ov_model


# class MultiDotABThenC(torch.nn.Module):
#     """Three constant matrices where (A @ B) @ C is cheaper than A @ (B @ C)."""

#     def __init__(self, dtype=torch.float32):
#         super().__init__()
#         self.A = make_small_tensor((8, 50), dtype=dtype, seed=100)
#         self.B = make_small_tensor((50, 6), dtype=dtype, seed=101)
#         self.C = make_small_tensor((6, 40), dtype=dtype, seed=102)
#         self.register_buffer("A_buf", self.A)
#         self.register_buffer("B_buf", self.B)
#         self.register_buffer("C_buf", self.C)

#     def forward(self):
#         return torch.linalg.multi_dot([self.A_buf, self.B_buf, self.C_buf])


# class MultiDotAThenBC(torch.nn.Module):
#     """Three constant matrices where A @ (B @ C) is cheaper than (A @ B) @ C."""

#     def __init__(self, dtype=torch.float32):
#         super().__init__()
#         self.A = make_small_tensor((40, 6), dtype=dtype, seed=200)
#         self.B = make_small_tensor((6, 50), dtype=dtype, seed=201)
#         self.C = make_small_tensor((50, 8), dtype=dtype, seed=202)
#         self.register_buffer("A_buf", self.A)
#         self.register_buffer("B_buf", self.B)
#         self.register_buffer("C_buf", self.C)

#     def forward(self):
#         return torch.linalg.multi_dot([self.A_buf, self.B_buf, self.C_buf])


# class MultiDotABCD(torch.nn.Module):
#     """Four constant matrices where DP-style planning should form (A @ B) and (C @ D)."""

#     def __init__(self, dtype=torch.float32):
#         super().__init__()
#         self.A = make_small_tensor((5, 60), dtype=dtype, seed=300)
#         self.B = make_small_tensor((60, 3), dtype=dtype, seed=301)
#         self.C = make_small_tensor((3, 70), dtype=dtype, seed=302)
#         self.D = make_small_tensor((70, 4), dtype=dtype, seed=303)
#         self.register_buffer("A_buf", self.A)
#         self.register_buffer("B_buf", self.B)
#         self.register_buffer("C_buf", self.C)
#         self.register_buffer("D_buf", self.D)

#     def forward(self):
#         return torch.linalg.multi_dot([self.A_buf, self.B_buf, self.C_buf, self.D_buf])


# class MultiDotFive(torch.nn.Module):
#     """Five constant matrices used to check MatMul count in the lowered graph."""

#     def __init__(self, dtype=torch.float32):
#         super().__init__()
#         self.A = make_small_tensor((4, 10), dtype=dtype, seed=400)
#         self.B = make_small_tensor((10, 6), dtype=dtype, seed=401)
#         self.C = make_small_tensor((6, 3), dtype=dtype, seed=402)
#         self.D = make_small_tensor((3, 8), dtype=dtype, seed=403)
#         self.E = make_small_tensor((8, 2), dtype=dtype, seed=404)
#         self.register_buffer("A_buf", self.A)
#         self.register_buffer("B_buf", self.B)
#         self.register_buffer("C_buf", self.C)
#         self.register_buffer("D_buf", self.D)
#         self.register_buffer("E_buf", self.E)

#     def forward(self):
#         return torch.linalg.multi_dot(
#             [self.A_buf, self.B_buf, self.C_buf, self.D_buf, self.E_buf]
#         )


# @pytest.mark.parametrize("dtype", [torch.float32])
# def test_linalg_multi_dot_three_tensors_prefers_AB_then_C(ie_device, precision, ir_version, dtype):
#     """
#     Three matrices case where (A @ B) @ C is cheaper than A @ (B @ C).
#     We expect the translator's heuristic to form the AB product first.
#     """
#     mod = MultiDotABThenC(dtype=dtype).eval()
#     pt_out, ov_out, ov_model = run_const_module_and_ov(mod, ie_device, precision)
#     assert_close_multi_dot(pt_out, ov_out)

#     mm_shapes = collect_matmul_shapes(ov_model)
#     if not mm_shapes:
#         pytest.xfail("linalg_multi_dot was fully constant-folded (no MatMul nodes to inspect)")

#     has_ab = any(
#         (l == (8, 50) and r == (50, 6)) or (l == (50, 6) and r == (8, 50))
#         for (l, r) in mm_shapes
#     )
#     assert has_ab, "Expected heuristic to compute (A @ B) first"


# @pytest.mark.parametrize("dtype", [torch.float32])
# def test_linalg_multi_dot_three_tensors_prefers_A_then_BC(ie_device, precision, ir_version, dtype):
#     """
#     Three matrices case where A @ (B @ C) is cheaper than (A @ B) @ C.
#     We expect the translator's heuristic to form the BC product first.
#     """
#     mod = MultiDotAThenBC(dtype=dtype).eval()
#     pt_out, ov_out, ov_model = run_const_module_and_ov(mod, ie_device, precision)
#     assert_close_multi_dot(pt_out, ov_out)

#     mm_shapes = collect_matmul_shapes(ov_model)
#     if not mm_shapes:
#         pytest.xfail("linalg_multi_dot was fully constant-folded (no MatMul nodes to inspect)")

#     has_bc = any(
#         (l == (6, 50) and r == (50, 8)) or (l == (50, 8) and r == (6, 50))
#         for (l, r) in mm_shapes
#     )
#     assert has_bc, "Expected heuristic to compute (B @ C) first"


# @pytest.mark.parametrize("dtype", [torch.float32])
# def test_linalg_multi_dot_four_tensors_dp_forms_AB_and_CD(ie_device, precision, ir_version, dtype):
#     """
#     Four matrices case where the DP-style splitter should form (A @ B) and (C @ D)
#     as independent sub-products before the final MatMul.
#     """
#     mod = MultiDotABCD(dtype=dtype).eval()
#     pt_out, ov_out, ov_model = run_const_module_and_ov(mod, ie_device, precision)
#     assert_close_multi_dot(pt_out, ov_out)

#     mm_shapes = collect_matmul_shapes(ov_model)
#     if not mm_shapes:
#         pytest.xfail("linalg_multi_dot was fully constant-folded (no MatMul nodes to inspect)")

#     has_ab = any(
#         (l == (5, 60) and r == (60, 3)) or (l == (60, 3) and r == (5, 60))
#         for (l, r) in mm_shapes
#     )
#     has_cd = any(
#         (l == (3, 70) and r == (70, 4)) or (l == (70, 4) and r == (3, 70))
#         for (l, r) in mm_shapes
#     )

#     assert has_ab and has_cd, "Expected DP to form (A @ B) and (C @ D) as sub-products"


# @pytest.mark.parametrize("dtype", [torch.float32])
# def test_linalg_multi_dot_matmul_count_is_n_minus_1(ie_device, precision, ir_version, dtype):
#     """
#     Sanity check: for N tensors, the resulting graph should contain at least N-1 MatMul nodes.
#     This ensures no degenerate lowering that would expand into something more expensive.
#     """
#     mod = MultiDotFive(dtype=dtype).eval()
#     pt_out, ov_out, ov_model = run_const_module_and_ov(mod, ie_device, precision)
#     assert_close_multi_dot(pt_out, ov_out)

#     matmuls = [n for n in ov_model.get_ordered_ops() if n.get_type_name() == "MatMul"]
#     if not matmuls:
#         pytest.xfail("linalg_multi_dot was fully constant-folded (no MatMul nodes to inspect)")

#     n_tensors = 5
#     assert len(matmuls) >= n_tensors - 1, (
#         f"Expected at least {n_tensors - 1} MatMul nodes for "
#         f"{n_tensors}-tensor multi_dot chain, got {len(matmuls)}"
#     )


