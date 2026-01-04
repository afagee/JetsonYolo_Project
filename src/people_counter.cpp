#include "../include/people_counter.hpp"
#include <algorithm>
#include <cmath>

// Constructor cập nhật line_x
PeopleCounter::PeopleCounter(int line_x, int max_disappeared, float max_distance)
    : line_x_(line_x), max_disappeared_(max_disappeared), max_distance_(max_distance),
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
    
    // Nếu không có detection nào, xử lý track biến mất
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
    
    // Match detections với existing tracks (Logic khoảng cách giữ nguyên)
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
    
    // Tạo tracks mới
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
        
        // --- LOGIC MỚI CHO LINE DỌC (KIỂM TRA TRỤC X) ---
        // Bên trái line: x < line_x
        // Bên phải line: x > line_x
        
        bool prev_left = track.prev_center_x < line_x_;
        bool curr_left = track.center_x < line_x_;
        
        if (prev_left != curr_left) {
            track.counted = true;
            
            // Từ Trái sang Phải -> Tính là VÀO (IN)
            if (prev_left && !curr_left) {
                count_in_++;
            } 
            // Từ Phải sang Trái -> Tính là RA (OUT)
            else if (!prev_left && curr_left) {
                count_out_++;
            }
            // Lưu ý: Nếu muốn đảo ngược logic Vào/Ra, chỉ cần đổi chỗ count_in_ và count_out_ ở trên
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
    // --- VẼ ĐƯỜNG ĐẾM DỌC ---
    const cv::Scalar red(0, 0, 255);
    // Vẽ từ (line_x, 0) đến (line_x, img.rows)
    cv::line(img, cv::Point(line_x_, 0), cv::Point(line_x_, img.rows), red, 2);
    
    // Vẽ chữ "Counting Line" xoay dọc hoặc để ngang cạnh line
    cv::putText(img, "Line", cv::Point(line_x_ + 5, 20),
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
    
    // Vẽ thông tin đếm (Góc trái trên)
    const int text_y = 30, text_height = 25;
    // Vẽ nền đen mờ cho text dễ đọc
    cv::rectangle(img, cv::Point(10, text_y - 20), 
                  cv::Point(200, text_y + text_height * 3), cv::Scalar(0, 0, 0), -1);
    
    cv::putText(img, "Vao (L->R): " + std::to_string(count_in_), cv::Point(15, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(img, "Ra (R->L): " + std::to_string(count_out_), cv::Point(15, text_y + text_height),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 165, 255), 2);
}
