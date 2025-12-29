#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <opencv2/opencv.hpp>

// Struct để lưu thông tin detection
struct Detection {
    float x1, y1, x2, y2;  // Tọa độ bounding box
    float confidence;      // Độ tin cậy
    int class_id;          // ID của class
};

// Struct để lưu thông tin box (alternative format)
struct Box {
    float x, y, w, h;      // Center x, center y, width, height
    float confidence;
    int class_id;
};

// Hàm tiện ích để vẽ bounding box lên ảnh
inline void drawBoundingBox(cv::Mat& img, const Detection& det, 
                            const std::vector<std::string>& class_names) {
    cv::Rect rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
    cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
    
    if (det.class_id >= 0 && det.class_id < static_cast<int>(class_names.size())) {
        std::string label = class_names[det.class_id] + ": " + 
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

#endif // COMMON_HPP

