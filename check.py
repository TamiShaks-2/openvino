
import openvino, inspect, os, glob
import openvino._pyopenvino as C
print("openvino package:", openvino.__file__)
c_dir = os.path.dirname(inspect.getfile(C))
print("C-extension dir:", c_dir)
cands = glob.glob(os.path.join(os.path.dirname(c_dir), "runtime", "lib", "intel64", "libopenvino_pytorch_frontend.so*"))
print("frontend .so candidates:", cands)

