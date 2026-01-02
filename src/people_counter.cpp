#include "../include/people_counter.hpp"
#include <algorithm>
#include <cmath>

PeopleCounter::PeopleCounter(int line_y, int max_disappeared, float max_distance)
    : line_y_(line_y), max_disappeared_(max_disappeared), max_distance_(max_distance),
      next_id_(1), count_in_(0), count_out_(0) {
}

float PeopleCounter::calculateDistance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

void PeopleCounter::updateTracks(const std::vector<Detection>& detections) {
    // Lọc chỉ lấy detections của người (class_id = 0)
    std::vector<Detection> people_detections;
    people_detections.reserve(detections.size());
    for (const auto& det : detections) {
        if (det.class_id == 0) {
            people_detections.push_back(det);
        }
    }
    
    // Nếu không có detection nào, tăng frames_since_seen cho tất cả tracks
    if (people_detections.empty()) {
        for (auto it = tracked_objects_.begin(); it != tracked_objects_.end();) {
            it->second.frames_since_seen++;
            if (it->second.frames_since_seen > max_disappeared_) {
                it = tracked_objects_.erase(it);
            } else {
                ++it;
            }
        }
        return;
    }
    
    // Tính center của mỗi detection
    std::vector<std::pair<float, float>> detection_centers;
    detection_centers.reserve(people_detections.size());
    for (const auto& det : people_detections) {
        detection_centers.emplace_back((det.x1 + det.x2) * 0.5f, (det.y1 + det.y2) * 0.5f);
    }
    
    // Match detections với existing tracks
    std::vector<bool> used_detections(people_detections.size(), false);
    
    for (auto& pair : tracked_objects_) {
        TrackedPerson& track = pair.second;
        float min_dist = max_distance_;
        int best_match = -1;
        
        for (size_t i = 0; i < detection_centers.size(); ++i) {
            if (used_detections[i]) continue;
            
            float dist = calculateDistance(
                track.center_x, track.center_y,
                detection_centers[i].first, detection_centers[i].second
            );
            
            if (dist < min_dist) {
                min_dist = dist;
                best_match = i;
            }
        }
        
        if (best_match != -1) {
            track.prev_center_x = track.center_x;
            track.prev_center_y = track.center_y;
            track.center_x = detection_centers[best_match].first;
            track.center_y = detection_centers[best_match].second;
            track.frames_since_seen = 0;
            used_detections[best_match] = true;
        } else {
            track.frames_since_seen++;
        }
    }
    
    // Xóa các tracks đã mất quá lâu
    for (auto it = tracked_objects_.begin(); it != tracked_objects_.end();) {
        if (it->second.frames_since_seen > max_disappeared_) {
            it = tracked_objects_.erase(it);
        } else {
            ++it;
        }
    }
    
    // Tạo tracks mới cho các detections chưa được match
    for (size_t i = 0; i < people_detections.size(); ++i) {
        if (!used_detections[i]) {
            int new_id = next_id_++;
            tracked_objects_.emplace(new_id, TrackedPerson(
                new_id,
                detection_centers[i].first,
                detection_centers[i].second
            ));
        }
    }
}

void PeopleCounter::checkCrossing() {
    for (auto& pair : tracked_objects_) {
        TrackedPerson& track = pair.second;
        if (track.counted) continue;
        
        bool prev_above = track.prev_center_y < line_y_;
        bool curr_above = track.center_y < line_y_;
        
        if (prev_above != curr_above) {
            track.counted = true;
            if (prev_above && !curr_above) {
                count_in_++;
            } else if (!prev_above && curr_above) {
                count_out_++;
            }
        }
    }
}

void PeopleCounter::update(const std::vector<Detection>& detections, int img_width, int img_height) {
    updateTracks(detections);
    checkCrossing();
}

void PeopleCounter::reset() {
    tracked_objects_.clear();
    count_in_ = 0;
    count_out_ = 0;
    next_id_ = 1;
}

void PeopleCounter::draw(cv::Mat& img) {
    // Vẽ đường đếm
    const cv::Scalar red(0, 0, 255);
    cv::line(img, cv::Point(0, line_y_), cv::Point(img.cols, line_y_), red, 2);
    cv::putText(img, "Counting Line", cv::Point(10, line_y_ - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, red, 2);
    
    // Vẽ các tracks
    const cv::Scalar blue(255, 0, 0), yellow(255, 255, 0);
    for (const auto& pair : tracked_objects_) {
        const TrackedPerson& track = pair.second;
        cv::Point center((int)track.center_x, (int)track.center_y);
        
        cv::circle(img, center, 5, blue, -1);
        cv::putText(img, "ID:" + std::to_string(track.id), 
                   cv::Point(center.x + 10, center.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, blue, 2);
        
        if (track.frames_since_seen == 0) {
            cv::line(img, cv::Point((int)track.prev_center_x, (int)track.prev_center_y),
                    center, yellow, 2);
        }
    }
    
    // Vẽ thông tin đếm
    const int text_y = 30, text_height = 25;
    cv::rectangle(img, cv::Point(10, text_y - 20), 
                 cv::Point(250, text_y + text_height * 3), cv::Scalar(0, 0, 0), -1);
    
    cv::putText(img, "Vao: " + std::to_string(count_in_), cv::Point(15, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(img, "Ra: " + std::to_string(count_out_), cv::Point(15, text_y + text_height),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2);
    cv::putText(img, "Tong: " + std::to_string(count_in_ - count_out_), 
                cv::Point(15, text_y + text_height * 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
}

