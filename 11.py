import onnx

model = onnx.load("yolo26n.onnx")
onnx.checker.check_model(model)
print("ONNX model check: OK")