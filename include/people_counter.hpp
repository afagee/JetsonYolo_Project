#ifndef PEOPLE_COUNTER_HPP
#define PEOPLE_COUNTER_HPP

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "yolov5.hpp"

struct TrackedPerson {
    int id;
    float center_x;
    float center_y;
    float prev_center_x;
    float prev_center_y;
    int frames_since_seen;
    bool counted;
    
    TrackedPerson(int id, float cx, float cy) 
        : id(id), center_x(cx), center_y(cy), 
          prev_center_x(cx), prev_center_y(cy),
          frames_since_seen(0), counted(false) {}
};

class PeopleCounter {
public:
    // Constructor nhận line_x thay vì line_y
    PeopleCounter(int line_x, int max_disappeared = 10, float max_distance = 100.0f);
    
    void update(const std::vector<Detection>& detections, int img_width, int img_height);
    void draw(cv::Mat& img);
    
    int getCountIn() const { return count_in_; }
    int getCountOut() const { return count_out_; }
    
    // Đổi tên hàm set/get cho trục X
    void setLineX(int x) { line_x_ = x; }
    int getLineX() const { return line_x_; }
    
    void reset();

private:
    int line_x_;  // Vị trí đường đếm (x coordinate - trục dọc)
    int max_disappeared_;
    float max_distance_;
    
    int next_id_;
    std::map<int, TrackedPerson> tracked_objects_;
    
    int count_in_;
    int count_out_;
    
    float calculateDistance(float x1, float y1, float x2, float y2);
    void updateTracks(const std::vector<Detection>& detections);
    void checkCrossing(); // Logic kiểm tra cắt ngang trục dọc
};

#endif // PEOPLE_COUNTER_HPP
