#include "yolo.h"
#include <chrono>
#include <numeric>

int main() {
    // === 配置区域 ===
    std::string enginePath = "models/yolov26n.engine";
    std::string videoPath = "test_video.mp4"; // 确保视频文件在同一目录
    std::string outputPath = "result_jetson_trt.mp4";
    // ================

    // 1. 初始化引擎
    std::cout << "🚀 初始化 TensorRT 引擎..." << std::endl;
    YoloInfer detector(enginePath);

    // 2. 打开视频
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "❌ 无法打开视频: " << videoPath << std::endl;
        return -1;
    }

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps_video = cap.get(cv::CAP_PROP_FPS);
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps_video, cv::Size(width, height));

    std::cout << "🎥 开始 TensorRT 推理 (NMS-Free)..." << std::endl;

    // 统计变量
    std::vector<double> instant_fps_list;
    auto overall_start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    cv::Mat frame;
    while (cap.read(frame)) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // === 核心推理 ===
        auto detections = detector.infer(frame);
        // ================

        auto t_end = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t_end - t_start).count(); // 秒
        double fps = 1.0 / dt;
        instant_fps_list.push_back(fps);
        frame_count++;

        // 绘制结果
        //for (const auto& det : detections) {
        //   cv::rectangle(frame, cv::Point(det[0], det[1]), cv::Point(det[2], det[3]), cv::Scalar(0, 255, 0), 2);
        //    // 这里可以添加文字标签，为了性能先简化
        //}

        // 绘制 FPS
        double avg_fps_display = 0;
        if (!instant_fps_list.empty()) {
            // 取最近10帧平均
            int n = std::min((int)instant_fps_list.size(), 10);
            avg_fps_display = std::accumulate(instant_fps_list.end() - n, instant_fps_list.end(), 0.0) / n;
        }

        //cv::rectangle(frame, cv::Point(10, 10), cv::Point(400, 60), cv::Scalar(0, 0, 0), -1);
        //cv::putText(frame, "TensorRT (Orin): " + std::to_string((int)avg_fps_display) + " FPS", 
        //            cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);

        //writer.write(frame);
        // cv::imshow("Jetson TensorRT", frame); // 在 Jetson 上如果是 SSH 连接，建议注释掉这行
        // if (cv::waitKey(1) == 'q') break;
    }

    auto overall_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(overall_end - overall_start).count();

    // 计算统计数据
    double avg_fps_all = frame_count / total_time;
    double mean_instant_fps = std::accumulate(instant_fps_list.begin(), instant_fps_list.end(), 0.0) / instant_fps_list.size();

    // === 打印最终报告 (与 Python 版对齐) ===
    std::cout << "\n==============================" << std::endl;
    std::cout << "4.1 Jetson Orin TensorRT (C++) 推理" << std::endl;
    std::cout << "Total frames      : " << frame_count << std::endl;
    std::cout << "Total time (s)    : " << total_time << std::endl;
    std::cout << "Average FPS (all) : " << avg_fps_all << std::endl;
    std::cout << "Mean instant FPS  : " << mean_instant_fps << std::endl;
    std::cout << "==============================\n" << std::endl;
    std::cout << "✅ 结果已保存: " << outputPath << std::endl;

    return 0;
}
