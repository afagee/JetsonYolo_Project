#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../include/yolov5.hpp"
#include "../include/people_counter.hpp"

inline bool exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <input_video> [--count]" << std::endl;
        std::cerr << "  --count  : Enable people counting feature (optional)" << std::endl;
        return -1;
    }

    std::string engine_file = argv[1];
    std::string input_video = argv[2];
    
    // Kiểm tra flag --count hoặc -c để bật tính năng đếm người
    bool enable_counting = false;
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--count" || arg == "-c") {
            enable_counting = true;
            break;
        }
    }

    if (!exists(engine_file) || !exists(input_video)) {
        std::cerr << "Error: File not found." << std::endl;
        return -1;
    }

    YOLOv5 yolo(engine_file);
    std::cout << "Engine loaded successfully!" << std::endl;

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file." << std::endl;
        return -1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    if (fps <= 0) fps = 30;

    std::string output_file = "result.avi";
    cv::VideoWriter writer(output_file, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot create output video writer." << std::endl;
        return -1;
    }
    std::cout << "Output: " << output_file << " (MJPG codec)" << std::endl;

    cv::Mat img;
    int frame_count = 0;
    
    // Khởi tạo PeopleCounter chỉ khi tính năng đếm người được bật
    std::unique_ptr<PeopleCounter> people_counter;
    int counting_line_y = height / 2;
    if (enable_counting) {
        people_counter = std::make_unique<PeopleCounter>(counting_line_y, 10, 100.0f);
        std::cout << "People counting enabled!" << std::endl;
        std::cout << "Controls: 'r'=reset, 'u'/'d'=move line" << std::endl;
    }
    
    std::cout << "Starting detection loop... (Press 'q' or ESC to quit)" << std::endl;

    // Tạo cửa sổ để hiển thị video
    cv::namedWindow("YOLO Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLO Detection", width, height);

    while (true) {
        cap >> img;
        if (img.empty()) {
            std::cout << "End of video." << std::endl;
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> detections = yolo.detect(img);

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double current_fps = 1000.0 / duration;

        // Update và vẽ people counter (chỉ khi được bật)
        if (enable_counting) {
            people_counter->update(detections, img.cols, img.rows);
        }

        // Vẽ detections
        yolo.drawDetections(img, detections);
        
        // Vẽ people counter
        if (enable_counting) {
            people_counter->draw(img);
        }
        
        // Vẽ thông tin FPS và số detection
        int info_y = enable_counting ? (img.rows - 100) : 30;
        const cv::Scalar text_color(0, 255, 0);
        cv::putText(img, "FPS: " + std::to_string(current_fps).substr(0, 4), 
                   cv::Point(10, info_y), cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);
        cv::putText(img, "Detections: " + std::to_string(detections.size()), 
                   cv::Point(10, info_y + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);
        cv::putText(img, "Frame: " + std::to_string(frame_count), 
                   cv::Point(10, info_y + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);

        frame_count++;
        if (frame_count % 10 == 0) { 
            std::cout << "Frame " << frame_count << " - " << duration << "ms - " 
                      << current_fps << " FPS - Detections: " << detections.size() << std::endl;
        }

        cv::imshow("YOLO Detection", img);
        writer.write(img);

        // Xử lý phím nhấn
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            break;
        } else if (enable_counting) {
            if (key == 'r' || key == 'R') {
                people_counter->reset();
                std::cout << "Counter reset!" << std::endl;
            } else if (key == 'u' || key == 'U') {
                counting_line_y = std::max(50, counting_line_y - 20);
                people_counter->setLineY(counting_line_y);
            } else if (key == 'd' || key == 'D') {
                counting_line_y = std::min(height - 50, counting_line_y + 20);
                people_counter->setLineY(counting_line_y);
            }
        }
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    std::cout << "Done! File saved: " << output_file << std::endl;
    return 0;
}
