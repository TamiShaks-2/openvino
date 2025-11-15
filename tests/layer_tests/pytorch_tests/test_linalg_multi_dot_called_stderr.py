import torch
import pytest
import openvino as ov

class MultiDotConstModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("A", torch.tensor([[1., 2.],
                                                [3., 4.]], dtype=torch.float32))
        self.register_buffer("B", torch.tensor([[1., 0.],
                                                [0., 1.]], dtype=torch.float32))
        self.register_buffer("C", torch.tensor([[5.],
                                                [6.]], dtype=torch.float32))
    def forward(self):
        return torch.linalg.multi_dot([self.A, self.B, self.C])

def build_frozen():
    m = MultiDotConstModule().eval()
    return torch.jit.freeze(torch.jit.script(m))

@pytest.mark.parametrize("dummy", [0])
def test_translate_multi_dot_printed_to_stderr(ie_device, precision, ir_version,
                                               dummy, capfd, monkeypatch):
    # מדליקים את דגל הדיבוג ב־C++ ומכבים טלמטריה כדי למנוע רעש/קריסות בסוף
    monkeypatch.setenv("OV_FE_PT_DEBUG", "1")
    monkeypatch.setenv("OPENVINO_TELEMETRY", "0")
    monkeypatch.setenv("OV_TELEMETRY", "0")
    monkeypatch.setenv("OV_DISABLE_TELEMETRY", "1")

    frozen = build_frozen()
    _ = ov.convert_model(frozen)  # פה ++C translate_multi_dot אמור לרוץ ולהדפיס ל-stderr

    out, err = capfd.readouterr()
    assert "[PT-FE] translate_multi_dot CALLED" in err, \
        "לא נתפסה ההדפסה מה-++C → כנראה translate_multi_dot לא רץ"
