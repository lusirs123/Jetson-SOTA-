import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO

# ================= 配置区域 =================
MODEL_PATH = 'yolo26n.pt'       # 建议暂时用 v8n 或 v11n 测试，确保能跑通
VIDEO_PATH = 'test_video.mp4'   # 刚才下载的视频
OUTPUT_PATH = 'result_mac_cpu.mp4'
DEVICE = 'cpu'                  # 强制使用 CPU 作为基准 或者使用mps
# ===========================================

def main():
    print(f"🚀 Loading model {MODEL_PATH} on {DEVICE.upper()}...")
    
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {VIDEO_PATH}")
        return

    # 视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (width, height))

    print(f"🎥 开始推理 (共 {total_frames_in_video} 帧)... 按 'q' 键提前结束")

    # --- 统计变量 ---
    frame_count = 0
    instant_fps_list = []  # 存储每一帧的瞬时 FPS
    
    # 记录总开始时间
    overall_start_time = time.time()

    while True:
        # 记录单帧开始时间
        frame_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        # --- 推理 ---
        # verbose=False 关闭库自带的打印，保持终端清爽
        results = model(frame, device=DEVICE, verbose=False) 
        
        # --- 后处理与绘图 ---
        annotated_frame = results[0].plot()

        # --- 时间计算 ---
        frame_end_time = time.time()
        process_time = frame_end_time - frame_start_time
        
        # 计算瞬时 FPS (防止除以0)
        instant_fps = 1.0 / process_time if process_time > 0 else 0.0
        instant_fps_list.append(instant_fps)
        frame_count += 1

        # 画面上显示的 FPS (取最近 10 帧平均，看起来更稳)
        display_fps = np.mean(instant_fps_list[-10:]) if len(instant_fps_list) > 10 else instant_fps
        
        # 绘制 UI
        cv2.rectangle(annotated_frame, (10, 10), (350, 60), (0, 0, 0), -1)
        cv2.putText(annotated_frame, 
                    f"CPU FPS: {display_fps:.1f}", 
                    (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 0, 255), 3)

        out.write(annotated_frame)
        cv2.imshow('YOLO Benchmark (Mac CPU)', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 用户手动停止")
            break

    # --- 最终统计计算 ---
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    
    # Average FPS (all) = 总帧数 / 总挂钟时间 (包含读取、推理、绘图、显示所有开销)
    avg_fps_all = frame_count / total_time if total_time > 0 else 0
    
    # Mean instant FPS = 所有单帧瞬时 FPS 的平均值 (更偏向推理性能)
    mean_instant_fps = np.mean(instant_fps_list) if instant_fps_list else 0

    # 清理
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- 打印要求的格式输出 ---
    print("\n" + "="*30)
    print(f"Total frames      : {frame_count}")
    print(f"Total time (s)    : {total_time:.1f}")
    print(f"Average FPS (all) : {avg_fps_all:.2f}")
    print(f"Mean instant FPS  : {mean_instant_fps:.2f}")
    print("="*30 + "\n")
    print(f"✅ 结果视频已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()