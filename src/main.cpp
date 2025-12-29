#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "../include/yolov5.hpp"

int main(int argc, char* argv[]) {
    // Parse arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <video_path> [conf_threshold] [nms_threshold]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../models/yolov5n.engine ../data/test_video.mp4 0.5 0.4" << std::endl;
        return 1;
    }
    
    std::string engine_path = argv[1];
    std::string video_path = argv[2];
    float conf_threshold = argc > 3 ? std::stof(argv[3]) : 0.5f;
    float nms_threshold = argc > 4 ? std::stof(argv[4]) : 0.4f;
    
    // Khởi tạo YOLOv5
    std::cout << "Loading YOLOv5 engine from: " << engine_path << std::endl;
    YOLOv5 yolo(engine_path, conf_threshold, nms_threshold);
    std::cout << "Engine loaded successfully!" << std::endl;
    
    // Mở video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << std::endl;
        return 1;
    }
    
    // Lấy thông tin video
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "Video info: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;
    
    // Tạo VideoWriter để lưu kết quả (optional)
    cv::VideoWriter writer;
    std::string output_path = "output_result.mp4";
    writer.open(output_path, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), fps, cv::Size(width, height));
    
    // Biến để tính FPS
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;
        
        // Inference
        auto inference_start = std::chrono::steady_clock::now();
        std::vector<Detection> detections = yolo.detect(frame);
        auto inference_end = std::chrono::steady_clock::now();
        
        // Tính inference time
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            inference_end - inference_start).count();
        float inference_fps = 1000.0f / inference_time;
        
        // Vẽ kết quả
        yolo.drawDetections(frame, detections);
        
        // Hiển thị thông tin FPS và số lượng detections
        std::string fps_text = "FPS: " + std::to_string(inference_fps).substr(0, 4);
        std::string det_text = "Detections: " + std::to_string(detections.size());
        
        cv::putText(frame, fps_text, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, det_text, cv::Point(10, 70),
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        
        // Hiển thị frame
        cv::imshow("YOLOv5 Detection", frame);
        
        // Lưu frame vào video output
        writer.write(frame);
        
        // Tăng frame count
        frame_count++;
        
        // Nhấn 'q' để thoát
        if (cv::waitKey(1) & 0xFF == 'q') {
            break;
        }
    }
    
    // Tính tổng FPS trung bình
    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    float avg_fps = frame_count / static_cast<float>(total_time);
    
    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << "Total frames processed: " << frame_count << std::endl;
    std::cout << "Total time: " << total_time << " seconds" << std::endl;
    std::cout << "Average FPS: " << avg_fps << std::endl;
    std::cout << "Output saved to: " << output_path << std::endl;
    
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    
    return 0;
}

