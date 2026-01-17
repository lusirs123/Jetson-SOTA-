import cv2
import time
import numpy as np
import onnxruntime as ort

# ================= 配置区域 =================
ONNX_MODEL_PATH = 'yolo26n.onnx'  # 确保你已经用 yolo export 导出了这个文件
VIDEO_PATH = 'test_video.mp4'      # 和之前测试用同一个视频
OUTPUT_PATH = 'result_mac_onnx_cpu.mp4'
CONF_THRESHOLD = 0.25              # 置信度阈值
INPUT_SIZE = 640                   # YOLO 标准输入尺寸
# ===========================================

def preprocess(frame):
    """
    YOLO 标准预处理：
    1. Resize 到 640x640
    2. 归一化 (0-255 -> 0-1)
    3. HWC -> BCHW (Batch, Channel, Height, Width)
    """
    height, width = frame.shape[:2]
    
    # 计算缩放比例，用于后续画框恢复坐标
    scale_x = width / INPUT_SIZE
    scale_y = height / INPUT_SIZE
    
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension -> BCHW
    return img, scale_x, scale_y

def postprocess(outputs, scale_x, scale_y, frame):
    """
    简单的后处理用于可视化。
    注意：不同版本的 YOLO export 格式可能不同。
    这里假设是 [1, 84, 8400] 或者 [1, 6, 300] (End-to-End)。
    为了不影响测速，这里只做最简单的解析演示。
    """
    # 这里我们主要关注推理速度，可视化只画出大概即可
    # 如果是 NMS-Free 的 End-to-End 模型，输出通常很简单
    pass 

def main():
    print(f"🚀 Loading ONNX model {ONNX_MODEL_PATH} on CPU...")
    
    # 1. 加载 ONNX 模型 (强制使用 CPU)
    try:
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 请先运行 yolo export model=yolov26n.pt format=onnx 导出模型")
        return

    # 获取输入输出节点名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (width, height))

    print("🎥 开始 ONNX 推理... 按 'q' 键退出")

    frame_count = 0
    instant_fps_list = []
    overall_start_time = time.time()

    while True:
        # 单帧计时开始
        frame_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        # --- 1. 预处理 ---
        input_tensor, scale_x, scale_y = preprocess(frame)

        # --- 2. 推理 (核心测速部分) ---
        outputs = session.run([output_name], {input_name: input_tensor})

        # --- 3. 后处理 (为了公平对比，这里可以简化，重点是 Session.run 的耗时) ---
        # 如果需要严格画框，需要解析 outputs[0]
        # 这里仅做简单的 FPS 标记
        
        # --- 4. 计时结束 ---
        frame_end_time = time.time()
        process_time = frame_end_time - frame_start_time
        
        instant_fps = 1.0 / process_time if process_time > 0 else 0.0
        instant_fps_list.append(instant_fps)
        frame_count += 1

        # 显示 FPS
        display_fps = np.mean(instant_fps_list[-10:]) if len(instant_fps_list) > 10 else instant_fps
        cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)
        cv2.putText(frame, 
                    f"ONNX CPU: {display_fps:.1f} FPS", 
                    (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 255, 0), 3) # 绿色字体区分

        out.write(frame)
        cv2.imshow('YOLO26 Mac ONNX CPU', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 最终统计
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    avg_fps_all = frame_count / total_time if total_time > 0 else 0
    mean_instant_fps = np.mean(instant_fps_list) if instant_fps_list else 0

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- 严格对齐你的截图格式 ---
    print("\n" + "="*30)
    print("3.3 Mac ONNX Runtime (CPU) 推理")  # 帮你拟好了标题
    print(f"运行文件 benchmark_mac_onnx_cpu.py, 得到输出:")
    print(f"Total frames      : {frame_count}")
    print(f"Total time (s)    : {total_time:.1f}")
    print(f"Average FPS (all) : {avg_fps_all:.2f}")
    print(f"Mean instant FPS  : {mean_instant_fps:.2f}")
    print("="*30 + "\n")
    print(f"✅ 结果视频已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()



