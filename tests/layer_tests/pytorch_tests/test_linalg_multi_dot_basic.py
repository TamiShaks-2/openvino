# test_linalg_multi_dot_basic.py
import torch
import openvino as ov
import numpy as np

# === Step 1: PyTorch reference ===
a = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
b = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
c = torch.tensor([[1.0],
                  [2.0]])

ref = torch.linalg.multi_dot([a, b, c])
print("🔹 PyTorch result:\n", ref.numpy())

# === Step 2: Export to TorchScript ===
class MultiDotModule(torch.nn.Module):
    def forward(self, x, y, z):
        return torch.linalg.multi_dot([x, y, z])

scripted = torch.jit.script(MultiDotModule())
scripted.save("multi_dot.pt")

# === Step 3: Convert to OpenVINO ===
core = ov.Core()
model = core.read_model("multi_dot.pt")
compiled_model = core.compile_model(model, "CPU")

# === Step 4: Run inference ===
inputs = {
    "x": a.numpy(),
    "y": b.numpy(),
    "z": c.numpy()
}
result = compiled_model(inputs)[compiled_model.output(0)]

print("🔹 OpenVINO result:\n", result)

# === Step 5: Compare numerically ===
np.testing.assert_allclose(result, ref.numpy(), rtol=1e-5, atol=1e-6)
print("✅ Test passed! Results match exactly.")
