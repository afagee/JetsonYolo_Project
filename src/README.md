# Source Directory

Thư mục này chứa mã nguồn C++ và CUDA của dự án.

## 📁 Nội dung

- `main.cpp`: File chính, chứa hàm `main()` và logic xử lý video
- `yolov5.cpp`: Cài đặt class `YOLOv5` - xử lý TensorRT inference
- `people_counter.cpp`: Cài đặt class `PeopleCounter` - logic đếm người vào/ra
- `yololayer.cu`: CUDA kernel cho YOLO layer (post-processing)

## 📝 Chi tiết từng file

### 1. main.cpp

**Chức năng chính**:
- Đọc tham số dòng lệnh (engine file, video path, flags)
- Khởi tạo YOLOv5 detector
- Đọc và xử lý video frame-by-frame
- Gọi detection và people counting (nếu được bật)
- Hiển thị kết quả và ghi video output
- Xử lý input từ bàn phím (điều khiển)

**Các tính năng**:
- Hỗ trợ flag `--count` hoặc `-c` để bật tính năng đếm người
- Tính và hiển thị FPS
- Ghi video output với tên `result.avi` (resize 50% để tối ưu)
- Điều khiển bằng bàn phím:
  - `q` hoặc `ESC`: Thoát
  - `r`: Reset counter (khi bật `--count`)
  - `l`: Di chuyển đường đếm sang trái (khi bật `--count`)
  - `k`: Di chuyển đường đếm sang phải (khi bật `--count`)

**Dependencies**:
- OpenCV (đọc/ghi video, hiển thị)
- YOLOv5 class (detection)
- PeopleCounter class (counting, nếu bật)

### 2. yolov5.cpp

**Chức năng chính**:
- Load TensorRT engine từ file
- Preprocess ảnh input (resize, normalize)
- Chạy inference trên GPU
- Post-process kết quả (decode boxes, NMS)
- Vẽ bounding boxes và labels lên ảnh

**Các phương thức chính**:
- `YOLOv5(engine_path)`: Constructor, load engine
- `detect(img)`: Chạy detection và trả về danh sách detections
- `drawDetections(img, detections)`: Vẽ bounding boxes và labels

**Dependencies**:
- TensorRT (NvInfer.h)
- CUDA (memory management)
- OpenCV (image processing)

### 3. people_counter.cpp

**Chức năng chính**:
- Tracking người qua các frame (distance-based matching)
- Phát hiện khi người vượt qua đường đếm
- Phân biệt hướng đi vào và đi ra
- Vẽ visualization (đường đếm, tracks, thống kê)

**Các phương thức chính**:
- `PeopleCounter(line_x, max_disappeared, max_distance)`: Constructor
- `update(detections, img_width, img_height)`: Cập nhật tracks và đếm
- `draw(img)`: Vẽ đường đếm, tracks, và thống kê
- `reset()`: Reset counter về 0
- `setLineX(x)`: Di chuyển đường đếm

**Thuật toán**:
- **Tracking**: Sử dụng distance-based matching giữa detections và existing tracks
- **Counting**: Phát hiện khi center của người vượt qua đường đếm (trục X)
- **Direction**: So sánh vị trí trước và sau để xác định hướng

**Dependencies**:
- OpenCV (drawing)
- Detection struct từ yolov5.hpp

### 4. yololayer.cu

**Chức năng chính**:
- CUDA kernel cho YOLO layer post-processing
- Decode bounding boxes từ output của TensorRT
- Tối ưu hóa bằng GPU để tăng tốc

**Lưu ý**:
- File này chứa CUDA code, được biên dịch bởi NVCC
- Cần được link với TensorRT plugin library

## 🔨 Build

Các file trong thư mục này được biên dịch bởi CMake:

```bash
cd build
cmake ..
make -j4
```

CMake sẽ tự động tìm tất cả file `.cpp` và `.cu` trong thư mục `src/`.

## 📚 Cấu trúc code

```
main.cpp
  ├── YOLOv5 (detection)
  │   ├── Load engine
  │   ├── Preprocess
  │   ├── Inference (TensorRT)
  │   └── Postprocess
  │
  └── PeopleCounter (counting, optional)
      ├── Update tracks
      ├── Check crossing
      └── Draw visualization
```

## 🔧 Tùy chỉnh

### Thay đổi input size

Sửa trong `yolov5.cpp`:
- `input_w_` và `input_h_` (mặc định: 640x640)

### Thay đổi confidence threshold

Sửa trong constructor của `YOLOv5`:
```cpp
YOLOv5 yolo(engine_file, 0.4f, 0.5f);  // conf_threshold, nms_threshold
```

### Thay đổi đường đếm

Trong `main.cpp`, thay đổi vị trí khởi tạo:
```cpp
int counting_line_x = width / 2;  // Mặc định ở giữa
```

### Thay đổi output video size

Trong `main.cpp`:
```cpp
float scale = 0.5;  // Thay đổi scale (0.5 = 50%)
```

## ⚠️ Lưu ý

- File `.cu` cần được biên dịch bởi NVCC (CUDA compiler)
- Đảm bảo CUDA và TensorRT được cài đặt đúng
- Code được viết cho C++14 standard
- Tối ưu cho Jetson Nano (compute capability 5.3)

