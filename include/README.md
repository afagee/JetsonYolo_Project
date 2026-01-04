# Include Directory

Thư mục này chứa các file header (.h, .hpp) định nghĩa các class, struct, và utility functions.

## 📁 Nội dung

- `yolov5.hpp`: Khai báo class `YOLOv5` - TensorRT inference
- `people_counter.hpp`: Khai báo class `PeopleCounter` - đếm người vào/ra
- `common.hpp`: Các struct và định nghĩa chung
- `yololayer.h`: Header cho YOLO layer CUDA
- `cuda_utils.h`: Utility functions cho CUDA error checking
- `logging.h`: Logger cho TensorRT
- `macros.h`: Các macro định nghĩa

## 📝 Chi tiết từng file

### 1. yolov5.hpp

**Nội dung**:
- Struct `Detection`: Chứa thông tin một detection (bounding box, confidence, class_id)
- Class `YOLOv5`: Class chính để chạy YOLOv5 inference với TensorRT

**Struct Detection**:
```cpp
struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float confidence;       // Confidence score
    int class_id;           // Class ID (0-79 cho COCO)
};
```

**Class YOLOv5**:
- `YOLOv5(engine_path, conf_threshold, nms_threshold)`: Constructor
- `detect(img)`: Chạy detection và trả về vector<Detection>
- `drawDetections(img, detections)`: Vẽ bounding boxes và labels

**Dependencies**:
- TensorRT (NvInfer.h)
- OpenCV (cv::Mat)

### 2. people_counter.hpp

**Nội dung**:
- Struct `TrackedPerson`: Thông tin một người được track
- Class `PeopleCounter`: Class để đếm người vào/ra

**Struct TrackedPerson**:
```cpp
struct TrackedPerson {
    int id;                    // Unique ID
    float center_x, center_y;  // Current center position
    float prev_center_x, prev_center_y;  // Previous position
    int frames_since_seen;     // Frames since last detection
    bool counted;              // Đã được đếm chưa
};
```

**Class PeopleCounter**:
- `PeopleCounter(line_x, max_disappeared, max_distance)`: Constructor
- `update(detections, img_width, img_height)`: Cập nhật tracks và đếm
- `draw(img)`: Vẽ visualization
- `getCountIn()`, `getCountOut()`: Lấy số đếm
- `setLineX(x)`, `getLineX()`: Điều khiển đường đếm
- `reset()`: Reset counter

**Dependencies**:
- yolov5.hpp (Detection struct)
- OpenCV (cv::Mat)

### 3. common.hpp

**Nội dung**:
- Các struct và định nghĩa chung được sử dụng trong toàn bộ project
- Có thể chứa các utility structs khác ngoài Detection (nếu có)

**Lưu ý**: File này có thể được mở rộng để thêm các định nghĩa chung khác.

### 4. yololayer.h

**Nội dung**:
- Header cho YOLO layer CUDA implementation
- Định nghĩa các hàm CUDA kernel cho post-processing
- Được sử dụng bởi `yololayer.cu`

**Dependencies**:
- CUDA
- TensorRT

### 5. cuda_utils.h

**Nội dung**:
- Utility functions để kiểm tra lỗi CUDA
- Macro `CUDA_CHECK(call)` để kiểm tra lỗi CUDA runtime
- Hữu ích để debug các vấn đề về CUDA

**Ví dụ sử dụng**:
```cpp
CUDA_CHECK(cudaMalloc(&ptr, size));
```

### 6. logging.h

**Nội dung**:
- Logger class cho TensorRT
- TensorRT yêu cầu một logger để log các thông báo và cảnh báo
- Implement `nvinfer1::ILogger` interface

**Sử dụng**: Được sử dụng bởi TensorRT builder và runtime.

### 7. macros.h

**Nội dung**:
- Các macro định nghĩa chung
- Có thể chứa các macro tiện ích, constants, hoặc helper macros

## 🔗 Mối quan hệ giữa các file

```
yolov5.hpp
  ├── Detection struct (có thể được định nghĩa ở đây hoặc common.hpp)
  └── YOLOv5 class

people_counter.hpp
  ├── TrackedPerson struct
  ├── PeopleCounter class
  └── Sử dụng Detection từ yolov5.hpp

common.hpp
  └── Các định nghĩa chung (nếu có)

yololayer.h
  └── CUDA kernel declarations

cuda_utils.h
  └── CUDA error checking utilities

logging.h
  └── TensorRT logger

macros.h
  └── Common macros
```

## 📚 Cách sử dụng

Các file header này được include trong các file source tương ứng:

- `main.cpp` includes: `yolov5.hpp`, `people_counter.hpp`
- `yolov5.cpp` includes: `yolov5.hpp`, `common.hpp`, `cuda_utils.h`, `logging.h`
- `people_counter.cpp` includes: `people_counter.hpp`
- `yololayer.cu` includes: `yololayer.h`

## 🔧 Tùy chỉnh

### Thêm struct mới

Thêm vào `common.hpp` hoặc tạo file header mới nếu cần.

### Thay đổi Detection struct

Sửa trong `yolov5.hpp` hoặc `common.hpp` (tùy nơi định nghĩa).

### Thêm phương thức mới

Thêm declaration vào header file tương ứng, implementation vào file `.cpp`.

## ⚠️ Lưu ý

- Header guards (`#ifndef`, `#define`, `#endif`) được sử dụng để tránh multiple inclusion
- Đảm bảo thứ tự include đúng để tránh dependency issues
- Các header file nên chỉ chứa declarations, không chứa implementations (trừ inline functions)

