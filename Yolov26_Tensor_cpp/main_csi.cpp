#include "yolo.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <cmath>

// ================= ⚙️ 配置区域 =================
const std::string ENGINE_PATH = "models/yolov26n.engine";
const int CAMERA_INDEX = 0;             // CSI 摄像头 ID（Jetson 通常为 0）
const bool ENABLE_GUI = true;           // true=演示模式(显示画面); false=跑分模式(无头模式)
const bool SAVE_VIDEO = true;           // 是否保存结果视频
const std::string OUTPUT_PATH = "output_csi.mp4"; // 统一命名格式
const float CONF_THRESHOLD = 0.25f;     // 置信度阈值
const float NMS_IOU_THRESHOLD = 0.45f;  // NMS 阈值
const int MAX_DRAW_DETECTIONS = 50;     // 绘制上限
const int MIN_BOX_AREA = 100;           // 过滤噪点
// ===============================================

// GStreamer 管道生成器 (1280x720@30fps)
std::string make_csi_pipeline(int sensor_id, int width, int height, int fps) {
    return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) +
           " ! video/x-raw(memory:NVMM), width=" + std::to_string(width) +
           ", height=" + std::to_string(height) +
           ", framerate=" + std::to_string(fps) + "/1" +
           " ! nvvidconv flip-method=0 ! video/x-raw, format=BGRx" +
           " ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1";
}

// 自动寻找可用摄像头
int find_available_csi_sensor(int max_sensors, int width, int height, int fps) {
    for (int i = 0; i < max_sensors; ++i) {
        std::string probe = make_csi_pipeline(i, width, height, fps);
        cv::VideoCapture test(probe, cv::CAP_GSTREAMER);
        if (test.isOpened()) {
            cv::Mat tmp;
            test >> tmp;
            if (!tmp.empty()) return i;
        }
    }
    return -1;
}

// 辅助结构体
struct DrawDetection {
    cv::Rect box;
    float conf;
    int class_id;
};

// 坐标限制
int clamp_coord(int value, int lower, int upper) {
    return std::max(lower, std::min(value, upper));
}

// IoU 计算
float compute_iou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    int inter_area = inter.area();
    if (inter_area <= 0) return 0.0f;
    int union_area = a.area() + b.area() - inter_area;
    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

// 准备绘制数据 (过滤 + NMS)
std::vector<DrawDetection> prepare_detections(const std::vector<std::vector<float>>& detections, const cv::Size& frame_size) {
    std::vector<DrawDetection> candidates;
    for (const auto& det : detections) {
        if (det.size() < 6) continue;
        float conf = det[4];
        if (conf < CONF_THRESHOLD) continue;

        int left = clamp_coord(static_cast<int>(std::round(det[0])), 0, frame_size.width - 1);
        int top = clamp_coord(static_cast<int>(std::round(det[1])), 0, frame_size.height - 1);
        int right = clamp_coord(static_cast<int>(std::round(det[2])), 0, frame_size.width - 1);
        int bottom = clamp_coord(static_cast<int>(std::round(det[3])), 0, frame_size.height - 1);

        if (right <= left || bottom <= top) continue;
        cv::Rect box(cv::Point(left, top), cv::Point(right, bottom));
        if (box.area() < MIN_BOX_AREA) continue;

        candidates.push_back({box, conf, static_cast<int>(det[5])});
    }

    // Sort by confidence
    std::sort(candidates.begin(), candidates.end(), [](const DrawDetection& a, const DrawDetection& b) {
        return a.conf > b.conf;
    });

    // Simple NMS
    std::vector<DrawDetection> filtered;
    for (const auto& cand : candidates) {
        bool keep = true;
        for (const auto& kept : filtered) {
            if (cand.class_id == kept.class_id && compute_iou(cand.box, kept.box) > NMS_IOU_THRESHOLD) {
                keep = false;
                break;
            }
        }
        if (keep) {
            filtered.push_back(cand);
            if (static_cast<int>(filtered.size()) >= MAX_DRAW_DETECTIONS) break;
        }
    }
    return filtered;
}

int main() {
    // 1. 初始化引擎
    std::cout << "🚀 [Init] Loading TensorRT Engine: " << ENGINE_PATH << "..." << std::endl;
    YoloInfer detector(ENGINE_PATH);
    std::cout << "✅ [Init] Engine loaded successfully." << std::endl;

    // 2. 打开 CSI 摄像头
    const int cap_width = 1280; // 使用 720p 采集以获得更好画质
    const int cap_height = 720;
    const int cap_fps = 30;

    int sensor_id = CAMERA_INDEX;
    int detected_id = find_available_csi_sensor(4, cap_width, cap_height, cap_fps);
    if (detected_id >= 0) sensor_id = detected_id;

    std::string pipeline = make_csi_pipeline(sensor_id, cap_width, cap_height, cap_fps);
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "❌ [Error] Could not open CSI camera." << std::endl;
        return -1;
    }

    std::cout << "🎥 [Start] TensorRT Inference Loop..." << std::endl;

    // 统计变量
    std::vector<double> instant_fps_list;
    auto overall_start = std::chrono::high_resolution_clock::now();
    long long frame_count = 0;
    cv::Mat frame;
    cv::VideoWriter writer;

    // 初始化视频保存
    if (SAVE_VIDEO) {
        // 如果想要保存的视频小一点，可以在这里 resize，但为了演示清晰度保持原样
        writer.open(OUTPUT_PATH,
                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    cap_fps,
                    cv::Size(cap_width, cap_height));
    }

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "⚠️ [Warn] Empty frame captured." << std::endl;
            break;
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // === 核心推理 ===
        auto detections = detector.infer(frame);
        auto drawable_detections = prepare_detections(detections, frame.size());
        // ================

        auto t_end = std::chrono::high_resolution_clock::now();
        
        // 计算瞬时 FPS
        double dt = std::chrono::duration<double>(t_end - t_start).count();
        double fps = 1.0 / dt;
        instant_fps_list.push_back(fps);
        frame_count++;

        // 绘制逻辑
        if (ENABLE_GUI || SAVE_VIDEO) {
            for (const auto& det : drawable_detections) {
                cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
                std::string label = std::to_string(det.class_id) + " " + std::to_string(static_cast<int>(det.conf * 100)) + "%";
                cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }

            // 计算显示的平滑 FPS
            double avg_fps_display = 0;
            if (!instant_fps_list.empty()) {
                int n = std::min((int)instant_fps_list.size(), 30);
                avg_fps_display = std::accumulate(instant_fps_list.end() - n, instant_fps_list.end(), 0.0) / n;
            }

            // 绘制 FPS
            cv::rectangle(frame, cv::Point(10, 10), cv::Point(350, 60), cv::Scalar(0, 0, 0), -1);
            cv::putText(frame, "CSI TensorRT: " + std::to_string((int)avg_fps_display) + " FPS", 
                        cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            if (ENABLE_GUI) {
                cv::imshow("YOLO26 Jetson CSI", frame);
                if (cv::waitKey(1) == 'q') break;
            }
        }

        if (SAVE_VIDEO && writer.isOpened()) {
            writer.write(frame);
        }
        
        // 简单的终端进度条
        if (frame_count % 60 == 0) {
            std::cout << "." << std::flush;
        }
    }

    // === 最终统计计算 (严格对齐 image_07383b.png 格式) ===
    auto overall_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(overall_end - overall_start).count();
    
    // Average FPS (all)
    double avg_fps_all = frame_count / total_time;
    
    // Mean instant FPS
    double mean_instant_fps = 0;
    if(!instant_fps_list.empty()) {
        mean_instant_fps = std::accumulate(instant_fps_list.begin(), instant_fps_list.end(), 0.0) / instant_fps_list.size();
    }

    if (SAVE_VIDEO && writer.isOpened()) writer.release();
    cap.release();
    if(ENABLE_GUI) cv::destroyAllWindows();

    // === 打印标准格式报告 ===
    std::cout << "\n\n==============================" << std::endl;
    std::cout << "4.2 Jetson Orin TensorRT (CSI) 推理" << std::endl; // 标题对齐
    std::cout << "Total frames      : " << frame_count << std::endl;
    std::cout << "Total time (s)    : " << total_time << std::endl;
    std::cout << "Average FPS (all) : " << avg_fps_all << std::endl;
    std::cout << "Mean instant FPS  : " << mean_instant_fps << std::endl;
    std::cout << "==============================" << std::endl;
    
    if (SAVE_VIDEO) {
        std::cout << "✅ 结果已保存: " << OUTPUT_PATH << std::endl;
    }

    return 0;
}