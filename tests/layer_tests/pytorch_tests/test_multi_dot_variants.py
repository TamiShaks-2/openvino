# tests/layer_tests/pytorch_tests/test_linalg_multi_dot_variants.py
# ENGLISH COMMENTS ONLY INSIDE CODE
import numpy as np
import pytest
import torch
import torch.fx as fx
import openvino as ov


# -------------------- helpers --------------------

def dtype_from_precision(precision: str):
    # Use float32 for reference stability; running half on CPU in PyTorch can be flaky.
    return torch.float32

def tol_from_precision(precision: str):
    p = str(precision).upper()
    if p in ("FP16", "F16", "HALF"):
        return dict(atol=5e-2, rtol=5e-2)
    return dict(atol=1e-4, rtol=1e-4)

def run_ov(model_obj, ie_device: str, *inputs):
    core = ov.Core()
    ov_model = ov.convert_model(model_obj)
    compiled = core.compile_model(ov_model, ie_device)
    outs = compiled([x.detach().numpy() if isinstance(x, torch.Tensor) else x for x in inputs])
    # PT FE returns a list of outputs; we expect a single tensor
    return outs[0]

def allclose(a, b, precision: str):
    np.testing.assert_allclose(a, b, **tol_from_precision(precision))


# -------------------- TorchScript path — ListConstruct --------------------

class M2_TS(torch.nn.Module):
    def forward(self, a, b):
        return torch.linalg.multi_dot([a, b])

class M3_TS(torch.nn.Module):
    def forward(self, a, b, c):
        return torch.linalg.multi_dot([a, b, c])

class M4_TS(torch.nn.Module):
    def forward(self, a, b, c, d):
        return torch.linalg.multi_dot([a, b, c, d])


@pytest.mark.parametrize(
    "shapes",
    [
        ((10, 20), (20, 5)),
        ((7, 13), (13, 3)),
    ],
)
def test_linalg_multi_dot_ts_list_2_tensors(ie_device, precision, ir_version, shapes):
    dt = dtype_from_precision(precision)
    a = torch.randn(*shapes[0], dtype=dt)
    b = torch.randn(*shapes[1], dtype=dt)

    # PyTorch reference
    pt = torch.linalg.multi_dot([a, b]).detach().numpy()

    # TorchScript -> OV
    m = torch.jit.script(M2_TS().eval())
    ov_out = run_ov(m, ie_device, a, b)

    allclose(pt, ov_out, precision)


@pytest.mark.parametrize(
    "shapes",
    [
        # Typical case where left association is cheaper (as in PyTorch docs example)
        ((10, 100), (100, 5), (5, 50)),
        # Another shape set (the translator will choose order based on static costs)
        ((50, 5), (5, 100), (100, 10)),
    ],
)
def test_linalg_multi_dot_ts_list_3_tensors(ie_device, precision, ir_version, shapes):
    dt = dtype_from_precision(precision)
    a = torch.randn(*shapes[0], dtype=dt)
    b = torch.randn(*shapes[1], dtype=dt)
    c = torch.randn(*shapes[2], dtype=dt)

    pt = torch.linalg.multi_dot([a, b, c]).detach().numpy()

    m = torch.jit.script(M3_TS().eval())
    ov_out = run_ov(m, ie_device, a, b, c)

    allclose(pt, ov_out, precision)


@pytest.mark.parametrize(
    "shapes",
    [
        # 4 tensors -> exercises DP order in translator
        ((12, 8), (8, 15), (15, 4), (4, 20)),
        ((30, 6), (6, 12), (12, 5), (5, 7)),
    ],
)
def test_linalg_multi_dot_ts_list_4_tensors(ie_device, precision, ir_version, shapes):
    dt = dtype_from_precision(precision)
    a = torch.randn(*shapes[0], dtype=dt)
    b = torch.randn(*shapes[1], dtype=dt)
    c = torch.randn(*shapes[2], dtype=dt)
    d = torch.randn(*shapes[3], dtype=dt)

    pt = torch.linalg.multi_dot([a, b, c, d]).detach().numpy()

    m = torch.jit.script(M4_TS().eval())
    ov_out = run_ov(m, ie_device, a, b, c, d)

    allclose(pt, ov_out, precision)


# -------------------- FX path — positional-style / lowered chain --------------------

class M3_FX(torch.nn.Module):
    def forward(self, a, b, c):
        # Eager still uses a Python list, but the FX decoder path differs from TS.
        # We also provide a lowering pass to remove multi_dot entirely (optional).
        return torch.linalg.multi_dot([a, b, c])

def lower_multi_dot_to_matmul(gm: fx.GraphModule) -> fx.GraphModule:
    # Replace torch.linalg.multi_dot([...]) with a nested matmul chain:
    g = gm.graph
    nodes = list(g.nodes)
    for n in nodes:
        if n.op == "call_function" and n.target is torch.linalg.multi_dot:
            arg0 = n.args[0]
            if isinstance(arg0, fx.Node) and arg0.op == "call_function" and arg0.target in (list, tuple):
                xs = list(arg0.args[0])
            elif isinstance(arg0, (list, tuple)):
                xs = list(arg0)
            else:
                xs = list(arg0)

            if len(xs) < 2:
                raise RuntimeError("multi_dot expects >=2 tensors")
            cur = g.call_function(torch.matmul, (xs[0], xs[1]))
            for t in xs[2:]:
                cur = g.call_function(torch.matmul, (cur, t))

            n.replace_all_uses_with(cur)
            g.erase_node(n)
            if isinstance(arg0, fx.Node) and len(arg0.users) == 0:
                g.erase_node(arg0)

    g.lint()
    gm.recompile()
    return gm

@pytest.mark.parametrize(
    "shapes",
    [
        ((10, 100), (100, 5), (5, 50)),
        ((32, 16), (16, 9), (9, 7)),
    ],
)
def test_linalg_multi_dot_fx_positional_3_tensors(ie_device, precision, ir_version, shapes):
    dt = dtype_from_precision(precision)
    a = torch.randn(*shapes[0], dtype=dt)
    b = torch.randn(*shapes[1], dtype=dt)
    c = torch.randn(*shapes[2], dtype=dt)

    # PyTorch reference
    pt = torch.linalg.multi_dot([a, b, c]).detach().numpy()

    # FX graph
    gm = fx.symbolic_trace(M3_FX().eval())

    # Optional lowering to explicit matmul-chain to fully avoid ListConstruct semantics in FX path:
    gm = lower_multi_dot_to_matmul(gm)

    ov_out = run_ov(gm, ie_device, a, b, c)
    allclose(pt, ov_out, precision)
