#include "yolo.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <cmath>

// ================= ⚙️ 配置区域 =================
const std::string ENGINE_PATH = "models/yolov26n.engine";
const int CAMERA_INDEX = 0;      // 摄像头 ID，通常为 0
const bool ENABLE_GUI = true;    // 【关键】true=演示模式(画框/显示); false=跑分模式(最快速度)
const bool SAVE_VIDEO = true;    // 是否保存检测视频
const std::string OUTPUT_PATH = "output_usb.mp4"; // 保存路径
const float CONF_THRESHOLD = 0.55f;     // 阈值略高一点，减少花屏
const float NMS_IOU_THRESHOLD = 0.45f;  // NMS 阈值
const int MAX_DRAW_DETECTIONS = 50;     // 每帧最多绘制 50 个目标，避免整屏遮挡
const int MIN_BOX_AREA = 100;           // 忽略非常小的框，滤掉噪声
// ===============================================

struct DrawDetection {
    cv::Rect box;
    float conf;
    int class_id;
};

int clamp_coord(int value, int lower, int upper) {
    if (upper < lower) return lower;
    return std::max(lower, std::min(value, upper));
}

float compute_iou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    int inter_area = inter.area();
    if (inter_area <= 0) return 0.0f;
    int union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0) return 0.0f;
    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

std::vector<DrawDetection> prepare_detections(const std::vector<std::vector<float>>& detections, const cv::Size& frame_size) {
    std::vector<DrawDetection> candidates;
    candidates.reserve(detections.size());

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

    std::sort(candidates.begin(), candidates.end(), [](const DrawDetection& a, const DrawDetection& b) {
        return a.conf > b.conf;
    });

    std::vector<DrawDetection> filtered;
    filtered.reserve(candidates.size());
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
    // 1. 初始化引擎 (匹配你的 main.cpp 构造方式)
    std::cout << "🚀 [Init] Loading TensorRT Engine: " << ENGINE_PATH << "..." << std::endl;
    YoloInfer detector(ENGINE_PATH);
    std::cout << "✅ [Init] Engine loaded successfully." << std::endl;

    // 2. 打开 USB 摄像头
    cv::VideoCapture cap(CAMERA_INDEX);
    if (!cap.isOpened()) {
        std::cerr << "❌ [Error] Could not open camera index " << CAMERA_INDEX << std::endl;
        return -1;
    }

    // 3. 设置摄像头参数 (降低分辨率以减少 CPU 预处理压力，释放更多算力给 NPU)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30); // 尝试请求 30 FPS

    std::cout << "🎥 [Start] TensorRT Inference Loop..." << std::endl;
    std::cout << "ℹ️  [Mode] GUI Visualization: " << (ENABLE_GUI ? "ON" : "OFF") << std::endl;

    // 统计变量
    std::vector<double> instant_fps_list;
    auto overall_start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    cv::Mat frame;
    cv::VideoWriter writer;

    if (SAVE_VIDEO) {
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps_out = cap.get(cv::CAP_PROP_FPS);
        if (fps_out <= 0) fps_out = 30.0; // 兜底
        writer.open(OUTPUT_PATH,
                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    fps_out,
                    cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "❌ [Error] Could not open video writer: " << OUTPUT_PATH << std::endl;
            return -1;
        }
    }

    while (true) {
        // 读取一帧
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "⚠️ [Warn] Empty frame captured." << std::endl;
            break;
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // === 核心推理 ===
        // 假设 yolo.h 返回的是 std::vector<Detection>
        auto detections = detector.infer(frame);
        auto drawable_detections = prepare_detections(detections, frame.size());
        // ================

        auto t_end = std::chrono::high_resolution_clock::now();
        
        // 计算时间
        double dt = std::chrono::duration<double>(t_end - t_start).count(); // 秒
        double fps = 1.0 / dt;
        instant_fps_list.push_back(fps);
        frame_count++;

        // === 分支 A: 演示模式 (绘制 + 显示) ===
        // 如果是为了给面试官看或者录像，开启此模式
        if (ENABLE_GUI) {
            for (const auto& det : drawable_detections) {
                cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

                std::string label = "ID:" + std::to_string(det.class_id) +
                                   " " + std::to_string(static_cast<int>(det.conf * 100)) + "%";
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                cv::Point labelTL(det.box.x, std::max(0, det.box.y - labelSize.height - baseLine));
                cv::Point labelBR(det.box.x + labelSize.width, det.box.y);
                cv::rectangle(frame, labelTL, labelBR, cv::Scalar(0, 255, 0), cv::FILLED);
                cv::putText(frame, label, cv::Point(det.box.x, det.box.y - baseLine),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }

            // 计算平滑 FPS 用于显示
            double avg_fps_display = 0;
            if (!instant_fps_list.empty()) {
                int n = std::min((int)instant_fps_list.size(), 20);
                avg_fps_display = std::accumulate(instant_fps_list.end() - n, instant_fps_list.end(), 0.0) / n;
            }

            // 显示 FPS
            cv::putText(frame, "TensorRT FPS: " + std::to_string((int)avg_fps_display), 
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            cv::imshow("YOLO26 Jetson USB", frame);
            //any key to exit
            if (cv::waitKey(1) == 'q') break;
        }
        if (SAVE_VIDEO && writer.isOpened()) {
            writer.write(frame);
        }
        // === 分支 B: 跑分模式 (仅终端输出) ===
        // 这里的 FPS 才是你写进 README 表格的真实性能
        else {
            if (frame_count % 100 == 0) {
                // 计算最近 100 帧的平均 FPS
                double current_avg = 0;
                int n = std::min((int)instant_fps_list.size(), 100);
                current_avg = std::accumulate(instant_fps_list.end() - n, instant_fps_list.end(), 0.0) / n;
                
                std::cout << "Processing... Frame: " << frame_count 
                          << " | Instant FPS: " << fps 
                          << " | Avg FPS (last 100): " << current_avg << std::endl;
            }
        }
    }

    // 最终统计
    auto overall_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(overall_end - overall_start).count();
    double avg_fps_all = frame_count / total_time;
    double mean_instant_fps = 0;
    if(!instant_fps_list.empty()) {
        mean_instant_fps = std::accumulate(instant_fps_list.begin(), instant_fps_list.end(), 0.0) / instant_fps_list.size();
    }

    // === 打印最终报告 (对齐你的格式) ===
    std::cout << "\n==========================================" << std::endl;
    std::cout << "4.2 Jetson Orin TensorRT (USB Camera) Report" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Mode              : " << (ENABLE_GUI ? "GUI (Slower)" : "Benchmark (Fastest)") << std::endl;
    std::cout << "Total frames      : " << frame_count << std::endl;
    std::cout << "Total time (s)    : " << total_time << std::endl;
    std::cout << "Average FPS (all) : " << avg_fps_all << " (Includes Camera I/O)" << std::endl;
    std::cout << "Mean Inference FPS: " << mean_instant_fps << " (Algorithm Only)" << std::endl;
    std::cout << "==========================================\n" << std::endl;

    if (SAVE_VIDEO && writer.isOpened()) writer.release();
    cap.release();
    if(ENABLE_GUI) cv::destroyAllWindows();

    return 0;
}