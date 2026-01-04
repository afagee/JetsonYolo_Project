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
        return -1;
    }

    std::string engine_file = argv[1];
    std::string input_video = argv[2];
    bool enable_counting = false;
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--count" || arg == "-c") enable_counting = true;
    }

    if (!exists(engine_file) || !exists(input_video)) {
        std::cerr << "Error: File not found." << std::endl;
        return -1;
    }

    YOLOv5 yolo(engine_file);
    std::cout << "Engine loaded." << std::endl;

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video." << std::endl;
        return -1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    if (fps <= 0) fps = 30;

    // --- CẤU HÌNH TỐI ƯU OUTPUT ---
    // Giảm kích thước video đầu ra để ghi nhanh hơn (giảm CPU Load)
    // Tỉ lệ 0.5 (giảm một nửa) hoặc cố định width = 640
    float scale = 0.5;
    cv::Size out_size(width * scale, height * scale);
    
    std::string output_file = "result.avi";
    // Lưu ý: Writer dùng out_size, không phải size gốc
    cv::VideoWriter writer(output_file, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, out_size);
    std::cout << "Output: " << output_file << " (Resized to " << out_size.width << "x" << out_size.height << ")" << std::endl;

    cv::Mat img, img_resized;
    int frame_count = 0;

    // Line dọc ở giữa
    int counting_line_x = width / 2;
    std::unique_ptr<PeopleCounter> people_counter;
    
    if (enable_counting) {
        people_counter = std::make_unique<PeopleCounter>(counting_line_x, 10, 100.0f);
        std::cout << "Counting Enabled. Controls: 'l'=Left, 'k'=Right, 'r'=Reset" << std::endl;
    }

    cv::namedWindow("YOLO Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLO Detection", width, height);

    while (true) {
        cap >> img;
        if (img.empty()) break;

        auto start = std::chrono::high_resolution_clock::now();

        // 1. Detect (Trên ảnh gốc để chính xác nhất)
        std::vector<Detection> detections = yolo.detect(img);

        // 2. Logic & Draw
        if (enable_counting) {
            people_counter->update(detections, img.cols, img.rows);
            people_counter->draw(img);
        }
        yolo.drawDetections(img, detections);

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double current_fps = 1000.0 / (duration > 0 ? duration : 1);

        // 3. Vẽ thông tin FPS (Góc Phải Trên)
        std::string fps_text = "FPS: " + std::to_string(current_fps).substr(0, 4);
        int baseLine = 0;
        cv::Size textSize = cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseLine);
        cv::putText(img, fps_text, cv::Point(width - textSize.width - 20, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        // 4. TỐI ƯU GHI VIDEO: Resize nhỏ lại trước khi ghi
        cv::resize(img, img_resized, out_size);
        writer.write(img_resized);

        // 5. Hiển thị
        cv::imshow("YOLO Detection", img);

        frame_count++;
        if (frame_count % 30 == 0) std::cout << "Frame " << frame_count << " | FPS: " << current_fps << std::endl;

        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
        
        if (enable_counting) {
            if (key == 'r') people_counter->reset();
            // Điều khiển Line sang Trái/Phải
            else if (key == 'l') { 
                counting_line_x = std::max(10, counting_line_x - 20);
                people_counter->setLineX(counting_line_x);
            }
            else if (key == 'k') { 
                counting_line_x = std::min(width - 10, counting_line_x + 20);
                people_counter->setLineX(counting_line_x);
            }
        }
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    std::cout << "Done." << std::endl;
    return 0;
}
