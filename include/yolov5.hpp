#ifndef YOLOV5_HPP
#define YOLOV5_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "common.hpp"

class YOLOv5 {
public:
    // Constructor: Load engine từ file .engine
    YOLOv5(const std::string& engine_path, float conf_threshold = 0.5f, float nms_threshold = 0.4f);
    
    // Destructor
    ~YOLOv5();
    
    // Hàm inference chính: Nhận ảnh đầu vào, trả về danh sách detections
    std::vector<Detection> detect(const cv::Mat& img);
    
    // Hàm để vẽ kết quả lên ảnh
    void drawDetections(cv::Mat& img, const std::vector<Detection>& detections);
    
    // Getter/Setter
    float getConfidenceThreshold() const { return conf_threshold_; }
    void setConfidenceThreshold(float threshold) { conf_threshold_ = threshold; }
    
    float getNMSThreshold() const { return nms_threshold_; }
    void setNMSThreshold(float threshold) { nms_threshold_ = threshold; }

private:
    // Khởi tạo TensorRT engine
    bool initEngine(const std::string& engine_path);
    
    // Preprocess: Resize và normalize ảnh
    void preprocess(const cv::Mat& img, float* data);
    
    // Postprocess: Parse output từ TensorRT
    std::vector<Detection> postprocess(float* output, int img_w, int img_h);
    
    // Non-maximum suppression
    std::vector<Detection> nms(const std::vector<Detection>& detections);
    
    // TensorRT members
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    // CUDA buffers
    void* input_buffer_;
    void* output_buffer_;
    
    // Model parameters
    int input_w_;
    int input_h_;
    int output_size_;
    float conf_threshold_;
    float nms_threshold_;
    
    // Class names (COCO dataset)
    std::vector<std::string> class_names_;
};

#endif // YOLOV5_HPP

