#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "../include/yolov5.hpp"
#include "../include/people_counter.hpp"

// --- CẤU TRÚC HÀNG ĐỢI AN TOÀN (THREAD-SAFE QUEUE) ---
template <typename T>
class SafeQueue {
private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cond;
    size_t max_size;

public:
    SafeQueue(size_t size) : max_size(size) {}

    // Đẩy frame vào queue (nếu đầy thì drop frame cũ nhất để giảm lag)
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex);
        if (queue.size() >= max_size) {
            queue.pop(); // Drop frame cũ nhất nếu xử lý không kịp
        }
        queue.push(item);
        lock.unlock();
        cond.notify_one();
    }

    // Lấy frame ra
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex);
        if (cond.wait_for(lock, std::chrono::milliseconds(5), [this] { return !queue.empty(); })) {
            item = std::move(queue.front());
            queue.pop();
            return true;
        }
        return false;
    }

    bool isEmpty() {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }
};

// Struct chứa dữ liệu cần hiển thị
struct FrameData {
    cv::Mat img;
    double fps;
};

// Biến toàn cục kiểm soát luồng
std::atomic<bool> g_running(true);

// --- HÀM XỬ LÝ HIỂN THỊ & GHI FILE (CHẠY LUỒNG RIÊNG) ---
void display_worker(SafeQueue<FrameData>& frame_queue, std::string output_file, double fps_stream, int width, int height) {
    // Chỉ khởi tạo VideoWriter nếu cần thiết
    float scale = 0.5;
    cv::Size out_size(width * scale, height * scale);
    cv::VideoWriter writer(output_file, cv::VideoWriter::fourcc('X', '2', '6', '4'), fps_stream, out_size);
    
    // Sử dụng OpenGL nếu OpenCV hỗ trợ (mượt hơn trên Jetson)
    try {
        cv::namedWindow("Jetson Nano YOLOv5", cv::WINDOW_OPENGL);
    } catch (...) {
        cv::namedWindow("Jetson Nano YOLOv5", cv::WINDOW_NORMAL);
    }

    cv::Mat img_resized;
    FrameData data;

    while (g_running) {
        if (frame_queue.pop(data)) {
            if (data.img.empty()) continue;

            // 1. Resize và Ghi file (Nặng - thực hiện ở đây để không chặn Inference)
            cv::resize(data.img, img_resized, out_size);
            writer.write(img_resized);

            // 2. Vẽ FPS lên ảnh gốc (đã có detection)
            cv::putText(data.img, "FPS: " + std::to_string(data.fps).substr(0, 4), 
                        cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            // 3. Hiển thị
            cv::imshow("Jetson Nano YOLOv5", data.img);

            // Xử lý phím bấm (nhẹ)
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {
                g_running = false;
            }
        }
    }
    writer.release();
    cv::destroyAllWindows();
}

// ... (Giữ nguyên các hàm helper exists và get_tegra_pipeline) ...
inline bool exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

std::string get_tegra_pipeline(int width, int height, int fps) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
           std::to_string(height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

std::string get_rtsp_pipeline(std::string rtsp_url, int latency=0) {
    return "rtspsrc location=" + rtsp_url + " latency=" + std::to_string(latency) + 
           " ! rtph264depay ! h264parse ! nvv4l2decoder ! "
           "nvvidconv ! video/x-raw, format=(string)BGRx ! "
           "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1";
}

int main(int argc, char** argv) {
    // ... (Giữ nguyên phần parse tham số đầu vào) ...
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <input_source> [--count] [output_file]" << std::endl;
        return -1;
    }
    std::string engine_file = argv[1];
    std::string input_source = argv[2];
    bool enable_counting = false;
    std::string output_file = "result.mp4";
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--count" || arg == "-c") enable_counting = true;
        else if (arg.find(".") != std::string::npos) output_file = arg;
    }

    if (!exists(engine_file)) {
        std::cerr << "Error: Engine file not found." << std::endl;
        return -1;
    }

    YOLOv5 yolo(engine_file);
    cv::VideoCapture cap;
    bool is_live = false;

    // ... (Giữ nguyên logic mở Camera/Video) ...
    if (input_source == "0") { cap.open(0); is_live = true; }
    else if (input_source == "csi") { cap.open(get_tegra_pipeline(1280, 720, 30), cv::CAP_GSTREAMER); is_live = true; }
    else if (input_source.find("rtsp://") != std::string::npos) { 
    // Dùng GStreamer thay vì FFMPEG
    cap.open(get_rtsp_pipeline(input_source, 0), cv::CAP_GSTREAMER); 
    is_live = true; 
}
    else { cap.open(input_source); }

    if (!cap.isOpened()) return -1;

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps_stream = is_live ? 30.0 : cap.get(cv::CAP_PROP_FPS);

    // Setup People Counter
    int counting_line_x = width / 2;
    std::unique_ptr<PeopleCounter> people_counter;
    if (enable_counting) {
        people_counter = std::make_unique<PeopleCounter>(counting_line_x, 10, 100.0f);
    }

    // --- KHỞI TẠO THREAD HIỂN THỊ ---
    // Giới hạn hàng đợi là 3 frame để tránh lag hình ảnh (latecy thấp)
    SafeQueue<FrameData> queue(3); 
    std::thread display_t(display_worker, std::ref(queue), output_file, fps_stream, width, height);
    display_t.detach(); // Cho chạy song song

    cv::Mat img;
    while (g_running) {
        cap >> img;
        if (img.empty()) {
            if (is_live) continue;
            else { g_running = false; break; }
        }

        auto start = std::chrono::high_resolution_clock::now();

        // 1. Inference (Đây là phần quan trọng nhất cần chạy nhanh)
        std::vector<Detection> detections = yolo.detect(img);

        // 2. Logic & Vẽ (Vẽ ngay trên luồng chính để đảm bảo sync với detection)
        if (enable_counting) {
            people_counter->update(detections, img.cols, img.rows);
            people_counter->draw(img);
        }
        yolo.drawDetections(img, detections);

        auto end = std::chrono::high_resolution_clock::now();
        double current_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 3. Đẩy sang luồng hiển thị
        // Clone ảnh để tránh xung đột vùng nhớ khi luồng chính tiếp tục đọc frame mới
        FrameData data = {img.clone(), current_fps}; 
        queue.push(data);
    }

    // Dọn dẹp
    g_running = false;
    if(display_t.joinable()) display_t.join();
    cap.release();
    return 0;
}