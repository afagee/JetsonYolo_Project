#include "../include/yolov5.hpp"
#include "../include/cuda_utils.h"
#include "../include/logging.h"
#include <fstream>
#include <algorithm>
#include <numeric>

// COCO class names
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
      input_buffer_(nullptr), output_buffer_(nullptr),
      input_w_(640), input_h_(640),
      conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
      class_names_(COCO_CLASSES) {
    
    if (!initEngine(engine_path)) {
        std::cerr << "Failed to initialize YOLOv5 engine!" << std::endl;
        exit(1);
    }
}

YOLOv5::~YOLOv5() {
    if (context_) {
        context_->destroy();
    }
    if (engine_) {
        engine_->destroy();
    }
    if (runtime_) {
        runtime_->destroy();
    }
    if (input_buffer_) {
        CUDA_CHECK(cudaFree(input_buffer_));
    }
    if (output_buffer_) {
        CUDA_CHECK(cudaFree(output_buffer_));
    }
}

bool YOLOv5::initEngine(const std::string& engine_path) {
    Logger logger;
    
    // Đọc file engine
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Cannot open engine file: " << engine_path << std::endl;
        return false;
    }
    
    // Đọc kích thước file
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Đọc dữ liệu engine
    char* trt_model_stream = new char[size];
    file.read(trt_model_stream, size);
    file.close();
    
    // Khởi tạo TensorRT runtime
    runtime_ = nvinfer1::createInferRuntime(logger);
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT runtime!" << std::endl;
        delete[] trt_model_stream;
        return false;
    }
    
    // Deserialize engine
    engine_ = runtime_->deserializeCudaEngine(trt_model_stream, size);
    delete[] trt_model_stream;
    
    if (!engine_) {
        std::cerr << "Failed to deserialize engine!" << std::endl;
        return false;
    }
    
    // Tạo execution context
    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "Failed to create execution context!" << std::endl;
        return false;
    }
    
    // Lấy thông tin input/output
    int input_index = engine_->getBindingIndex("images");
    int output_index = engine_->getBindingIndex("output");
    
    nvinfer1::Dims input_dims = engine_->getBindingDimensions(input_index);
    input_w_ = input_dims.d[3];
    input_h_ = input_dims.d[2];
    
    nvinfer1::Dims output_dims = engine_->getBindingDimensions(output_index);
    output_size_ = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<int>());
    
    // Cấp phát CUDA buffers
    size_t input_size = input_w_ * input_h_ * 3 * sizeof(float);
    size_t output_size = output_size_ * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&input_buffer_, input_size));
    CUDA_CHECK(cudaMalloc(&output_buffer_, output_size));
    
    return true;
}

void YOLOv5::preprocess(const cv::Mat& img, float* data) {
    cv::Mat resized, rgb;
    cv::resize(img, resized, cv::Size(input_w_, input_h_));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    // Normalize về [0, 1] và transpose từ HWC sang CHW
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_h_; ++h) {
            for (int w = 0; w < input_w_; ++w) {
                data[c * input_h_ * input_w_ + h * input_w_ + w] = 
                    rgb.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }
}

std::vector<Detection> YOLOv5::detect(const cv::Mat& img) {
    int img_w = img.cols;
    int img_h = img.rows;
    
    // Preprocess
    float* input_data = new float[input_w_ * input_h_ * 3];
    preprocess(img, input_data);
    
    // Copy input to GPU
    CUDA_CHECK(cudaMemcpy(input_buffer_, input_data, 
                          input_w_ * input_h_ * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));
    delete[] input_data;
    
    // Run inference
    void* bindings[] = {input_buffer_, output_buffer_};
    context_->executeV2(bindings);
    
    // Copy output from GPU
    float* output = new float[output_size_];
    CUDA_CHECK(cudaMemcpy(output, output_buffer_, 
                          output_size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    // Postprocess
    std::vector<Detection> detections = postprocess(output, img_w, img_h);
    delete[] output;
    
    // Apply NMS
    detections = nms(detections);
    
    return detections;
}

std::vector<Detection> YOLOv5::postprocess(float* output, int img_w, int img_h) {
    std::vector<Detection> detections;
    
    // Parse output (format: [batch, num_detections, 85] where 85 = x, y, w, h, conf, 80 classes)
    // Giả sử output là 1D array với shape [25200, 85]
    int num_detections = output_size_ / 85;
    
    float scale_x = static_cast<float>(img_w) / input_w_;
    float scale_y = static_cast<float>(img_h) / input_h_;
    
    for (int i = 0; i < num_detections; ++i) {
        float* det = output + i * 85;
        
        float x_center = det[0];
        float y_center = det[1];
        float width = det[2];
        float height = det[3];
        float conf = det[4];
        
        if (conf < conf_threshold_) continue;
        
        // Tìm class có confidence cao nhất
        float max_class_conf = 0.0f;
        int class_id = 0;
        for (int j = 5; j < 85; ++j) {
            if (det[j] > max_class_conf) {
                max_class_conf = det[j];
                class_id = j - 5;
            }
        }
        
        float final_conf = conf * max_class_conf;
        if (final_conf < conf_threshold_) continue;
        
        // Convert từ center format sang corner format
        float x1 = (x_center - width / 2.0f) * scale_x;
        float y1 = (y_center - height / 2.0f) * scale_y;
        float x2 = (x_center + width / 2.0f) * scale_x;
        float y2 = (y_center + height / 2.0f) * scale_y;
        
        Detection detection;
        detection.x1 = x1;
        detection.y1 = y1;
        detection.x2 = x2;
        detection.y2 = y2;
        detection.confidence = final_conf;
        detection.class_id = class_id;
        
        detections.push_back(detection);
    }
    
    return detections;
}

std::vector<Detection> YOLOv5::nms(const std::vector<Detection>& detections) {
    if (detections.empty()) return detections;
    
    // Sort by confidence (descending)
    std::vector<Detection> sorted_dets = detections;
    std::sort(sorted_dets.begin(), sorted_dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<Detection> result;
    std::vector<bool> suppressed(sorted_dets.size(), false);
    
    for (size_t i = 0; i < sorted_dets.size(); ++i) {
        if (suppressed[i]) continue;
        
        result.push_back(sorted_dets[i]);
        
        // Suppress overlapping boxes
        float area_i = (sorted_dets[i].x2 - sorted_dets[i].x1) * 
                       (sorted_dets[i].y2 - sorted_dets[i].y1);
        
        for (size_t j = i + 1; j < sorted_dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (sorted_dets[i].class_id != sorted_dets[j].class_id) continue;
            
            // Calculate IoU
            float x1 = std::max(sorted_dets[i].x1, sorted_dets[j].x1);
            float y1 = std::max(sorted_dets[i].y1, sorted_dets[j].y1);
            float x2 = std::min(sorted_dets[i].x2, sorted_dets[j].x2);
            float y2 = std::min(sorted_dets[i].y2, sorted_dets[j].y2);
            
            float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float area_j = (sorted_dets[j].x2 - sorted_dets[j].x1) * 
                           (sorted_dets[j].y2 - sorted_dets[j].y1);
            float iou = inter / (area_i + area_j - inter);
            
            if (iou > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

void YOLOv5::drawDetections(cv::Mat& img, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::Rect rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
        cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
        
        std::string label = class_names_[det.class_id] + ": " + 
                           std::to_string(det.confidence).substr(0, 4);
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(img, cv::Point(det.x1, det.y1 - text_size.height - 5),
                     cv::Point(det.x1 + text_size.width, det.y1),
                     cv::Scalar(0, 255, 0), -1);
        cv::putText(img, label, cv::Point(det.x1, det.y1 - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

