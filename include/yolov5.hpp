#ifndef YOLOV5_HPP
#define YOLOV5_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class YOLOv5 {
public:
    YOLOv5(const std::string& engine_path, float conf_threshold = 0.4f, float nms_threshold = 0.5f);
    ~YOLOv5();

    std::vector<Detection> detect(const cv::Mat& img);
    void drawDetections(cv::Mat& img, const std::vector<Detection>& detections);

private:
    bool initEngine(const std::string& engine_path);
    void preprocess(const cv::Mat& img);
    // Sửa dòng này: Nhận kích thước ảnh gốc để tính scale ngay lập tức
    std::vector<Detection> postprocess(int original_w, int original_h);
    std::vector<Detection> nms(const std::vector<Detection>& detections);

    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;

    void* input_buffer_device_;
    void* output_buffer_device_;
    float* h_input_data_;
    float* h_output_data_;

    int input_w_;
    int input_h_;
    int output_size_;
    float conf_threshold_;
    float nms_threshold_;
    std::vector<std::string> class_names_;
};

#endif // YOLOV5_HPP
