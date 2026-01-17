from ultralytics import YOLO

# 加载预训练模型（会自动下载）
model = YOLO("yolo26n.pt")  # 或 yolo26s.pt，根据你的显存选择

# 导出 ONNX（推荐参数）
success = model.export(
    format="onnx",      # 导出格式
    imgsz=640,          # 输入尺寸（YOLO26 检测默认 640）
    dynamic=False,       # 支持动态输入形状（batch/size 可变，加分项）
    simplify=True,      # 简化 ONNX 图，减少节点
    opset=18            # 最新 opset，兼容性最好
)

if success:
    print("ONNX 导出成功！文件：yolo26n.onnx")
    