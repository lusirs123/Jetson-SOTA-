#include "yolo.h"
#include <algorithm>

// 构造函数
YoloInfer::YoloInfer(const std::string& enginePath) {
    // 1. 读取 Engine 文件
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "❌ Error: 无法读取引擎文件: " << enginePath << std::endl;
        exit(1);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    // 2. 初始化 TensorRT
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    context = engine->createExecutionContext();
    delete[] trtModelStream;

    // 3. 分配 GPU 显存
    // 输入: 1 * 3 * 640 * 640
    cudaMalloc(&buffers[0], 3 * INPUT_W * INPUT_H * sizeof(float));
    
    // 输出: 1 * 84 * 8400
    // 根据你的 Netron 图，输出大小是 84 * 8400
    output_size = OUTPUT_CHANNELS * OUTPUT_CANDIDATES; 
    cudaMalloc(&buffers[1], output_size * sizeof(float));

    // 4. 分配 CPU 内存
    cpu_output_buffer = new float[output_size];
    std::cout << "✅ TensorRT 引擎加载成功！(适配 1x84x8400 + NMS)" << std::endl;
}

YoloInfer::~YoloInfer() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete[] cpu_output_buffer;
    delete context;
    delete engine;
    delete runtime;
}

// 预处理
// 替换 yolo.cpp 中的 preprocess 函数

void YoloInfer::preprocess(cv::Mat& img, float* gpu_input_buffer) {
    // 直接 Resize 到 640x640（不做 letterbox）
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, cv::Size(INPUT_W, INPUT_H), cv::Scalar(0, 0, 0), true, false);
    cudaMemcpy(gpu_input_buffer, blob.ptr<float>(), 3 * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice);
}

// 核心推理函数
std::vector<std::vector<float>> YoloInfer::infer(cv::Mat& img) {
    // 1. 预处理
    preprocess(img, (float*)buffers[0]);

    // 2. 推理
    context->executeV2(buffers);

    // 3. 拷贝结果回 CPU (1x84x8400)
    cudaMemcpy(cpu_output_buffer, buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 4. 后处理准备
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float* output = cpu_output_buffer;
    
    // === 关键修复：内存转置解析 ===
    // 你的模型输出是 [1, 84, 8400]
    // 内存排列：[Row 0: 8400个cx] [Row 1: 8400个cy] ... [Row 4: 8400个Class0概率] ...
    
    const int num_anchors = OUTPUT_CANDIDATES; // 8400

    auto sigmoid = [](float x) -> float {
        return 1.0f / (1.0f + std::exp(-x));
    };

    // 判断是否为归一化坐标 (0-1)
    bool bbox_normalized = true;
    int sample_count = std::min(200, num_anchors);
    for (int i = 0; i < sample_count; i++) {
        float v0 = output[0 * num_anchors + i];
        float v1 = output[1 * num_anchors + i];
        float v2 = output[2 * num_anchors + i];
        float v3 = output[3 * num_anchors + i];
        float vmax = std::max(std::max(std::fabs(v0), std::fabs(v1)), std::max(std::fabs(v2), std::fabs(v3)));
        if (vmax > 1.5f) {
            bbox_normalized = false;
            break;
        }
    }

    for (int i = 0; i < num_anchors; i++) {
        // 1. 找出当前 Anchor 在 80 个类别中的最大置信度
        // 类别概率从第 4 行开始 (0,1,2,3 是坐标)
        float max_score = -1.0f;
        int max_class_id = -1;

        // 遍历 80 个类别
        for (int c = 0; c < 80; c++) {
            // 访问逻辑：buffer[ 行号 * 8400 + 列号 ]
            float score = sigmoid(output[(4 + c) * num_anchors + i]);
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }

        if (max_score >= 0.45f) {
            float cx = output[0 * num_anchors + i];
            float cy = output[1 * num_anchors + i];
            float w  = output[2 * num_anchors + i];
            float h  = output[3 * num_anchors + i];

            if (bbox_normalized) {
                cx *= INPUT_W;
                cy *= INPUT_H;
                w *= INPUT_W;
                h *= INPUT_H;
            }

            float x_factor = static_cast<float>(img.cols) / INPUT_W;
            float y_factor = static_cast<float>(img.rows) / INPUT_H;
            float x1 = (cx - 0.5f * w) * x_factor;
            float y1 = (cy - 0.5f * h) * y_factor;
            float x2 = (cx + 0.5f * w) * x_factor;
            float y2 = (cy + 0.5f * h) * y_factor;

            int left = std::max(0, std::min(static_cast<int>(std::round(x1)), img.cols - 1));
            int top = std::max(0, std::min(static_cast<int>(std::round(y1)), img.rows - 1));
            int right = std::max(0, std::min(static_cast<int>(std::round(x2)), img.cols - 1));
            int bottom = std::max(0, std::min(static_cast<int>(std::round(y2)), img.rows - 1));
            if (right <= left || bottom <= top) continue;

            boxes.push_back(cv::Rect(cv::Point(left, top), cv::Point(right, bottom)));
            confidences.push_back(max_score);
            classIds.push_back(max_class_id);
        }
    }

    // 5. NMS (非极大值抑制) - 解决满屏绿框的关键
    std::vector<int> indices;
    if (!boxes.empty() && boxes.size() == confidences.size()) {
        // 参数：boxes, confidences, score_threshold, nms_threshold
        cv::dnn::NMSBoxes(boxes, confidences, 0.45f, 0.45f, indices);
    }

    // 6. 格式化输出为 vector<vector<float>> 以匹配 yolo.h
    std::vector<std::vector<float>> final_results;
    for (int idx : indices) {
        std::vector<float> det;
        det.push_back((float)boxes[idx].x);                     // x1
        det.push_back((float)boxes[idx].y);                     // y1
        det.push_back((float)(boxes[idx].x + boxes[idx].width)); // x2
        det.push_back((float)(boxes[idx].y + boxes[idx].height));// y2
        det.push_back(confidences[idx]);                        // conf
        det.push_back((float)classIds[idx]);                    // class_id
        final_results.push_back(det);
    }

    return final_results;
}
