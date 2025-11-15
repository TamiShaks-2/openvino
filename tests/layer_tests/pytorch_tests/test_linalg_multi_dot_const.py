import torch
import numpy as np
import pytest

class MultiDotConstModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # קבועים כ-buffers (לא פרמטרים), כדי ש-freeze יוכל לקבע אותם בגרף
        self.register_buffer("A", torch.tensor([[1.0, 2.0],
                                                [3.0, 4.0]], dtype=torch.float32))
        self.register_buffer("B", torch.tensor([[1.0, 0.0],
                                                [0.0, 1.0]], dtype=torch.float32))
        self.register_buffer("C", torch.tensor([[5.0],
                                                [6.0]], dtype=torch.float32))

    def forward(self):
        xs = [self.A, self.B, self.C]      # ← prim::ListConstruct של באפרים קבועים
        return torch.linalg.multi_dot(xs)  # ← aten::linalg_multi_dot

@pytest.mark.parametrize("dummy", [0])
def test_linalg_multi_dot_const_freeze(ie_device, precision, ir_version, dummy):
    mod = MultiDotConstModule().eval()
    scripted = torch.jit.script(mod)
    frozen = torch.jit.freeze(scripted)   # ← מקבע get_attr לקבועים בגרף כשאפשר

    # (אופציונלי לאבחון) – הדפסת הגרף כדי לוודא שיש Constants:
    # print(frozen.graph)

    # רפרנס מפייתון
    ref = torch.linalg.multi_dot([mod.A, mod.B, mod.C]).numpy()

    import openvino as ov
    ov_model = ov.convert_model(frozen)

    core = ov.Core()
    compiled = core.compile_model(ov_model, ie_device)
    out = next(iter(compiled.create_infer_request().infer().values()))

    if precision == "FP16":
        np.testing.assert_allclose(out.astype(np.float16), ref.astype(np.float16), rtol=2e-2, atol=2e-2)
    else:
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
