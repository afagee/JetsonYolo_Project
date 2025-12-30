#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "../include/yolov5.hpp"

inline bool exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <input_video>" << std::endl;
        return -1;
    }

    std::string engine_file = argv[1];
    std::string input_video = argv[2];

    if (!exists(engine_file) || !exists(input_video)) {
        std::cerr << "Error: File not found." << std::endl;
        return -1;
    }

    //std::cout << "1. Loading Engine..." << std::endl;
    YOLOv5 yolo(engine_file);
    std::cout << "Engine loaded successfully!" << std::endl;

    //std::cout << "2. Opening Video..." << std::endl;
    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file." << std::endl;
        return -1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    if (fps <= 0) fps = 30; // Fallback nếu không đọc được FPS

    //std::cout << "Video Info: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;

    // --- SỬA ĐỔI QUAN TRỌNG: Dùng .avi và MJPG (An toàn tuyệt đối) ---
    std::string output_file = "result.avi";
    cv::VideoWriter writer(output_file, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
    
    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot create output video writer." << std::endl;
        return -1;
    }
    std::cout << "Output set to: " << output_file << " (MJPG codec)" << std::endl;

    cv::Mat img;
    int frame_count = 0;
    
    std::cout << "3. Starting Loop..." << std::endl;

    while (true) {
        // STEP A: Read
	//std::cout<<"chet o day r1";
        cap >> img;
	//std::cout<<"chet o day r2";
        if (img.empty()) {
            std::cout << "End of video." << std::endl;
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // STEP B: Detect (In ra để xem có chết ở đây không)
        //std::cout << "DEBUG: Detecting..." << std::endl; 
        std::vector<Detection> detections = yolo.detect(img);

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // STEP C: Draw
        //std::cout << "DEBUG: Drawing..." << std::endl;
        yolo.drawDetections(img, detections);
	
	std::cout<<detections.size()<<std::endl;
        frame_count++;
        if (frame_count % 10 == 0) { 
             std::cout << "Frame " << frame_count << " - " << duration << "ms - " << (1000.0 / duration) << " FPS" << std::endl;
        }

        // STEP D: Write (Thường hay chết ở đây)
        // std::cout << "DEBUG: Writing..." << std::endl;
        writer.write(img);
    }

    cap.release();
    writer.release();
    std::cout << "Done! File saved: " << output_file << std::endl;

    return 0;
}
