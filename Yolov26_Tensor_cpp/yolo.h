#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

// 简单的 Logger
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) 
            std::cout << "[TRT] " << msg << std::endl;
    }
};

class YoloInfer {
public:
    YoloInfer(const std::string& enginePath);
    ~YoloInfer();

    // 返回格式: [x1, y1, x2, y2, confidence, class_id]
    std::vector<std::vector<float>> infer(cv::Mat& img);

private:
    Logger logger;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    void* buffers[2]; 
    float* cpu_output_buffer = nullptr; 

    // === 关键参数修改 ===
    const int INPUT_W = 640;
    const int INPUT_H = 640;
    const int NUM_CLASSES = 80;  // COCO 类别数
    const int NUM_ANCHORS = 8400; // YOLOv8/v11/v26 标准锚框数
    const int OUTPUT_CHANNELS = 84; // 4(coords) + 80(classes)
    const int OUTPUT_SIZE = OUTPUT_CHANNELS * NUM_ANCHORS; 

    void preprocess(cv::Mat& img, float* gpu_input_buffer, float& r, int& d_w, int& d_h);
};