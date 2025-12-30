#include "../include/yolov5.hpp"
#include "../include/cuda_utils.h"
#include "../include/logging.h"
#include <fstream>
#include <numeric>
#include <iostream>

static int g_input_index = -1;
static int g_output_index = -1;

static const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

YOLOv5::YOLOv5(const std::string& engine_path, float conf_threshold, float nms_threshold)
    : runtime_(nullptr), engine_(nullptr), context_(nullptr),
      input_buffer_device_(nullptr), output_buffer_device_(nullptr),
      h_input_data_(nullptr), h_output_data_(nullptr),
      input_w_(640), input_h_(640),
      conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
      class_names_(COCO_CLASSES) {
    if (!initEngine(engine_path)) {
        std::cerr << "FATAL: Init Engine Failed" << std::endl;
        exit(1);
    }
}

YOLOv5::~YOLOv5() {
    if (context_) context_->destroy();
    if (engine_) engine_->destroy();
    if (runtime_) runtime_->destroy();
    if (input_buffer_device_) cudaFree(input_buffer_device_);
    if (output_buffer_device_) cudaFree(output_buffer_device_);
    if (h_input_data_) delete[] h_input_data_;
    if (h_output_data_) delete[] h_output_data_;
}

bool YOLOv5::initEngine(const std::string& engine_path) {
    Logger logger;
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) return false;
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trt_model_stream = new char[size];
    file.read(trt_model_stream, size);
    file.close();

    runtime_ = nvinfer1::createInferRuntime(logger);
    engine_ = runtime_->deserializeCudaEngine(trt_model_stream, size);
    delete[] trt_model_stream;
    if (!engine_) return false;
    context_ = engine_->createExecutionContext();

    g_input_index = engine_->getBindingIndex("data");
    g_output_index = engine_->getBindingIndex("prob");
    if (g_input_index == -1) g_input_index = engine_->getBindingIndex("images");
    if (g_output_index == -1) g_output_index = engine_->getBindingIndex("output");
    if (g_input_index == -1 || g_output_index == -1) return false;

    nvinfer1::Dims input_dims = engine_->getBindingDimensions(g_input_index);
    if (input_dims.nbDims == 4) {
        input_h_ = input_dims.d[2]; input_w_ = input_dims.d[3];
    } else {
        input_h_ = input_dims.d[1]; input_w_ = input_dims.d[2];
    }

    nvinfer1::Dims output_dims = engine_->getBindingDimensions(g_output_index);
    output_size_ = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<int>());

    cudaMalloc(&input_buffer_device_, input_w_ * input_h_ * 3 * sizeof(float));
    cudaMalloc(&output_buffer_device_, output_size_ * sizeof(float));
    h_input_data_ = new float[input_w_ * input_h_ * 3];
    h_output_data_ = new float[output_size_];

    return true;
}

void YOLOv5::preprocess(const cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_w_, input_h_), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    
    int h = input_h_;
    int w = input_w_;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                h_input_data_[c * h * w + i * w + j] = resized.at<cv::Vec3b>(i, j)[c] / 255.0f;
            }
        }
    }
}

std::vector<Detection> YOLOv5::detect(const cv::Mat& img) {
    if (img.empty()) return {};

    preprocess(img);
    cudaMemcpy(input_buffer_device_, h_input_data_, input_w_ * input_h_ * 3 * sizeof(float), cudaMemcpyHostToDevice);

    void* bindings[2];
    bindings[g_input_index] = input_buffer_device_;
    bindings[g_output_index] = output_buffer_device_;
    
    context_->execute(1, bindings);

    cudaMemcpy(h_output_data_, output_buffer_device_, output_size_ * sizeof(float), cudaMemcpyDeviceToHost);

    // Truyền kích thước ảnh gốc vào đây để tính tỷ lệ chính xác
    return nms(postprocess(img.cols, img.rows));
}

// Hàm này đã được sửa lại giống hệt phiên bản Debug (Tính scale ngay lập tức)
std::vector<Detection> YOLOv5::postprocess(int img_w, int img_h) {
    std::vector<Detection> detections;
    int num_detections = output_size_ / 85;
    
    float scale_x = (float)img_w / input_w_;
    float scale_y = (float)img_h / input_h_;

    for (int i = 0; i < num_detections; ++i) {
        float* det = h_output_data_ + i * 85;
        float conf = det[4];
        
        // Chỉ bỏ qua nếu quá thấp (ví dụ < 10%)
        if (conf < 0.1f) continue;

        float max_class_conf = 0.0f;
        int class_id = -1;
        for (int j = 5; j < 85; ++j) {
            if (det[j] > max_class_conf) {
                max_class_conf = det[j];
                class_id = j - 5;
            }
        }
        
        float final_conf = conf * max_class_conf;
        if (final_conf < 0.1f) continue;

        // --- ĐOẠN DEBUG QUAN TRỌNG ---
        // In ra để biết nó nhìn thấy gì
        std::cout << "DEBUG: Found Class ID: " << class_id 
                  << " - Conf: " << final_conf 
                  << " (Person is ID 0)" << std::endl;
        
        // Tạm thời KHÔNG LỌC class_id != 0 nữa để xem nó ra cái gì
        // if (class_id != 0) continue; 
        // -----------------------------

        float x = det[0], y = det[1], w = det[2], h = det[3];
        Detection d;
        d.x1 = (x - w/2) * scale_x;
        d.y1 = (y - h/2) * scale_y;
        d.x2 = (x + w/2) * scale_x;
        d.y2 = (y + h/2) * scale_y;
        d.confidence = final_conf;
        d.class_id = class_id;
        detections.push_back(d);
    }
    return detections;
}
std::vector<Detection> YOLOv5::nms(const std::vector<Detection>& detections) {
    if (detections.empty()) return {};
    std::vector<Detection> sorted = detections;
    std::sort(sorted.begin(), sorted.end(), [](const Detection& a, const Detection& b){return a.confidence > b.confidence;});
    std::vector<Detection> res;
    res.reserve(detections.size());
    std::vector<bool> suppressed(sorted.size(), false);
    for(size_t i=0; i<sorted.size(); ++i){
        if(suppressed[i]) continue;
        res.push_back(sorted[i]);
        float area_i = (sorted[i].x2 - sorted[i].x1)*(sorted[i].y2 - sorted[i].y1);
        for(size_t j=i+1; j<sorted.size(); ++j){
            if(suppressed[j]) continue;
            if(sorted[i].class_id != sorted[j].class_id) continue;
            float x1 = std::max(sorted[i].x1, sorted[j].x1);
            float y1 = std::max(sorted[i].y1, sorted[j].y1);
            float x2 = std::min(sorted[i].x2, sorted[j].x2);
            float y2 = std::min(sorted[i].y2, sorted[j].y2);
            float w = std::max(0.0f, x2-x1);
            float h = std::max(0.0f, y2-y1);
            float inter = w*h;
            float iou = inter / (area_i + (sorted[j].x2-sorted[j].x1)*(sorted[j].y2-sorted[j].y1) - inter + 1e-6);
            if(iou > nms_threshold_) suppressed[j] = true;
        }
    }
    return res;
}

void YOLOv5::drawDetections(cv::Mat& img, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        // Tọa độ đã được scale ở postprocess rồi, giờ chỉ việc vẽ
        int x1 = std::max(0, (int)det.x1);
        int y1 = std::max(0, (int)det.y1);
        int x2 = std::min(img.cols-1, (int)det.x2);
        int y2 = std::min(img.rows-1, (int)det.y2);
        
        cv::rectangle(img, cv::Rect(x1, y1, x2-x1, y2-y1), cv::Scalar(0,255,0), 2);
    }
}
