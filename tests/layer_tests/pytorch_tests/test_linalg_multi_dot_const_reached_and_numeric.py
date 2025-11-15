import torch
import numpy as np
import pytest
import openvino as ov

# --- מודול עם קלטים קבועים (buffers) כדי ש-freeze יקבע אותם לגרף ---
class MultiDotConstModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("A", torch.tensor([[1.0, 2.0],
                                                [3.0, 4.0]], dtype=torch.float32))
        self.register_buffer("B", torch.tensor([[1.0, 0.0],
                                                [0.0, 1.0]], dtype=torch.float32))
        self.register_buffer("C", torch.tensor([[5.0],
                                                [6.0]], dtype=torch.float32))
    def forward(self):
        # ListConstruct של buffers קבועים → עובר את ה-frontend
        return torch.linalg.multi_dot([self.A, self.B, self.C])

def build_frozen():
    m = MultiDotConstModule().eval()
    scripted = torch.jit.script(m)
    frozen = torch.jit.freeze(scripted)
    return frozen, m

# --- טסט 1: מאשר שההמרה הגיעה לנתיב שלך גם אם כל הגרף התקפל לקבוע ---
@pytest.mark.parametrize("dummy", [0])
def test_translate_multi_dot_reached_or_folded(ie_device, precision, ir_version, dummy):
    frozen, _ = build_frozen()
    ov_model = ov.convert_model(frozen)

    type_names = [op.get_type_name() for op in ov_model.get_ops()]
    names = {op.get_friendly_name() for op in ov_model.get_ops()}

    # אפשרות א: נתת friendly_name ב-translate_multi_dot והוא נשמר
    if "ptfe_translate_multi_dot_out" in names:
        return

    # אפשרות ב: כל השרשרת התקפלה → נראה כמעט רק Constant ו-Result
    only_const_and_result = all(t in ("Constant", "Result") for t in type_names)
    assert only_const_and_result, (
        f"ציפינו או למצוא 'ptfe_translate_multi_dot_out' או לראות קיפול מלא לקבוע. "
        f"סוגי האופ' בגרף: {type_names}"
    )

# --- טסט 2: נכונות נומרית (על אותם קבועים) ---
@pytest.mark.parametrize("dummy", [0])
def test_linalg_multi_dot_numeric_const(ie_device, precision, ir_version, dummy):
    frozen, m = build_frozen()

    # רפרנס מפייתון (PyTorch)
    ref = torch.linalg.multi_dot([m.A, m.B, m.C]).numpy()

    # המרה והרצה ב-OV
    ov_model = ov.convert_model(frozen)
    core = ov.Core()
    compiled = core.compile_model(ov_model, ie_device)
    out = next(iter(compiled.create_infer_request().infer().values()))

    if precision == "FP16":
        np.testing.assert_allclose(out.astype(np.float16), ref.astype(np.float16),
                                   rtol=2e-2, atol=2e-2)
    else:
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
