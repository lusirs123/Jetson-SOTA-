#include "yolo.h"
#include <algorithm>

YoloInfer::YoloInfer(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary); // 修正后的构造
    if (!file.good()) {
        std::cerr << "❌ 无法读取引擎: " << enginePath << std::endl;
        abort();
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    context = engine->createExecutionContext();
    delete[] trtModelStream;

    cudaMalloc(&buffers[0], 3 * INPUT_W * INPUT_H * sizeof(float));
    cudaMalloc(&buffers[1], OUTPUT_SIZE * sizeof(float));
    cpu_output_buffer = new float[OUTPUT_SIZE];
}

YoloInfer::~YoloInfer() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete[] cpu_output_buffer;
    delete context;
    delete engine;
    delete runtime;
}

void YoloInfer::preprocess(cv::Mat& img, float* gpu_input_buffer, float& r, int& d_w, int& d_h) {
    int w = img.cols;
    int h = img.rows;
    float scale = std::min((float)INPUT_W / w, (float)INPUT_H / h);
    r = scale;
    int new_w = (int)(w * scale);
    int new_h = (int)(h * scale);
    d_w = (INPUT_W - new_w) / 2;
    d_h = (INPUT_H - new_h) / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    cv::Mat canvas(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(d_w, d_h, new_w, new_h)));

    // ✅ 关键：确保 BGR 转 RGB
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    canvas.convertTo(canvas, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(canvas, channels);

    float* host_input = new float[3 * INPUT_W * INPUT_H];
    for (int i = 0; i < 3; i++) {
        memcpy(host_input + i * INPUT_W * INPUT_H, channels[i].data, INPUT_W * INPUT_H * sizeof(float));
    }
    cudaMemcpy(gpu_input_buffer, host_input, 3 * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice);
    delete[] host_input;
}

std::vector<std::vector<float>> YoloInfer::infer(cv::Mat& img) {
    float scale;
    int d_w, d_h;
    preprocess(img, (float*)buffers[0], scale, d_w, d_h);
    context->executeV2(buffers);
    cudaMemcpy(cpu_output_buffer, buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;

    float* ptr = cpu_output_buffer;
    
    // ⬇️⬇️ 调试探针：记录这一帧看到的最高分 ⬇️⬇️
    float debug_max_score = 0.0f; 

    for (int i = 0; i < NUM_ANCHORS; ++i) {
        float max_score = -1.0f;
        int max_class = -1;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float score = ptr[(4 + c) * NUM_ANCHORS + i];
            if (score > max_score) {
                max_score = score;
                max_class = c;
            }
        }
        
        if (max_score > debug_max_score) debug_max_score = max_score;

        // ✅ 这里的阈值改为 0.25 (Nano 模型在逆光下分通常不高)
        if (max_score > 0.25) { 
            float cx = ptr[0 * NUM_ANCHORS + i];
            float cy = ptr[1 * NUM_ANCHORS + i];
            float w  = ptr[2 * NUM_ANCHORS + i];
            float h  = ptr[3 * NUM_ANCHORS + i];

            float r_cx = (cx - d_w) / scale;
            float r_cy = (cy - d_h) / scale;
            float r_w  = w / scale;
            float r_h  = h / scale;

            int left   = (int)(r_cx - r_w * 0.5f);
            int top    = (int)(r_cy - r_h * 0.5f);
            int width  = (int)r_w;
            int height = (int)r_h;

            left = std::max(0, left);
            top  = std::max(0, top);
            width = std::min(width, img.cols - left);
            height = std::min(height, img.rows - top);

            boxes.push_back(cv::Rect(left, top, width, height));
            confs.push_back(max_score);
            class_ids.push_back(max_class);
        }
    }

    // ⬇️⬇️ 如果最高分太低，在终端报警 ⬇️⬇️
    static int low_conf_counter = 0;
    if (debug_max_score < 0.25) {
        low_conf_counter++;
        if (low_conf_counter % 30 == 0) { // 每30帧提醒一次
             std::cout << "⚠️ 警告：当前画面最高置信度仅为 " << debug_max_score 
                       << "，可能是逆光或模型输入颜色不对！" << std::endl;
        }
    }

    // NMS
    std::vector<int> indices;
    // ✅ NMS 阈值也改为 0.25
    cv::dnn::NMSBoxes(boxes, confs, 0.25f, 0.45f, indices);

    std::vector<std::vector<float>> results;
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        results.push_back({
            (float)box.x, (float)box.y, 
            (float)(box.x + box.width), (float)(box.y + box.height), 
            confs[idx], (float)class_ids[idx]
        });
    }
    return results;
}