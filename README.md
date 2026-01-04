# JetsonYolo_Project

Dự án triển khai YOLOv5 với TensorRT trên NVIDIA Jetson Nano để tối ưu hóa hiệu suất inference. Dự án bao gồm tính năng **đếm người vào/ra** với tracking và hiển thị trực quan trên video.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Tính năng](#tính-năng)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🎯 Tổng quan

Dự án này triển khai YOLOv5 object detection với TensorRT trên Jetson Nano, cung cấp:

- **Object Detection**: Phát hiện đối tượng real-time với YOLOv5, hỗ trợ 80 classes từ COCO dataset
- **People Counting**: Tính năng đếm người vào/ra với tracking và visualization (tùy chọn)
- **Tối ưu hiệu suất**: Sử dụng TensorRT để đạt FPS cao trên Jetson Nano
- **Dễ sử dụng**: Hỗ trợ cả chế độ detection thuần túy và chế độ đếm người

## 📁 Cấu trúc dự án

```
JetsonYolo_Project/
├── CMakeLists.txt          # File cấu hình build
├── README.md               # Hướng dẫn sử dụng chính
├── .gitignore              # Loại bỏ file rác khi up lên Git
│
├── build/                  # Nơi chứa file thực thi sau khi biên dịch
│   └── README.md           # Hướng dẫn về thư mục build
│
├── data/                   # Chứa dữ liệu test
│   ├── README.md           # Hướng dẫn về dữ liệu
│   └── test_video.mp4      # Video test (không commit lên Git)
│
├── include/                # Chứa các file Header (.h, .hpp)
│   ├── README.md           # Mô tả các header files
│   ├── common.hpp          # Các struct chung (Detection, Box...)
│   ├── cuda_utils.h        # Hàm kiểm tra lỗi CUDA
│   ├── logging.h           # Logger bắt buộc của TensorRT
│   ├── macros.h            # Các macro định nghĩa
│   ├── people_counter.hpp  # Khai báo Class PeopleCounter
│   ├── yololayer.h         # Header cho YOLO layer CUDA
│   └── yolov5.hpp          # Khai báo Class YOLOv5
│
├── models/                 # Chứa weights và engine files
│   ├── README.md           # Hướng dẫn về models
│   ├── yolov5n.pt          # Model PyTorch gốc
│   ├── yolov5n.wts         # File trung gian (.pt -> .wts)
│   └── yolov5n.engine      # TensorRT engine (không commit lên Git)
│
├── scripts/                # Các file Python phụ trợ
│   ├── README.md           # Hướng dẫn về scripts
│   ├── gen_wts.py          # Convert .pt -> .wts
│   ├── build_engine.py     # Build TensorRT engine từ ONNX
│   └── compare_fps.py      # So sánh FPS với Python version
│
└── src/                    # Chứa mã nguồn C++/CUDA
    ├── README.md           # Mô tả source files
    ├── main.cpp            # Hàm main: Đọc video, gọi YOLO, tính FPS
    ├── people_counter.cpp  # Cài đặt Class PeopleCounter
    ├── yololayer.cu        # CUDA kernel cho YOLO layer
    └── yolov5.cpp          # Cài đặt Class YOLOv5
```

Chi tiết về từng thư mục, xem README.md trong từng thư mục.

## 🔧 Yêu cầu hệ thống

### Hardware
- **NVIDIA Jetson Nano** (hoặc các dòng Jetson khác)
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Storage**: Đủ dung lượng cho models và video

### Software
- **OS**: JetPack 4.6+ hoặc JetPack 5.x
- **CUDA**: 
  - 10.2+ (với JetPack 4.6)
  - 11.4+ (với JetPack 5.x)
- **TensorRT**: 8.0+
- **OpenCV**: 4.5+
- **CMake**: 3.10+
- **Python**: 3.6+ (cho scripts)

## 📦 Cài đặt

### 1. Cài đặt dependencies

```bash
# Cập nhật package list
sudo apt-get update

# Cài đặt OpenCV (nếu chưa có)
sudo apt-get install libopencv-dev

# Kiểm tra TensorRT
dpkg -l | grep tensorrt
```

TensorRT thường được cài đặt sẵn với JetPack.

### 2. Chuẩn bị model

#### Bước 2.1: Download YOLOv5 model

```bash
# Download YOLOv5n model (khuyến nghị cho Jetson Nano)
cd models
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt
```

#### Bước 2.2: Build TensorRT engine

Có 3 cách để build TensorRT engine:

**Cách 1: Build từ file .wts (Khuyến nghị cho Jetson Nano)**

```bash
# Clone tensorrtx repository
cd ~
git clone -b yolov5-v6.0 https://github.com/wang-xinyu/tensorrtx.git
cd tensorrtx/yolov5

# Copy file .wts vào thư mục này
cp /path/to/JetsonYolo_Project/models/yolov5n.wts .

# Build engine
mkdir build && cd build
cmake ..
make

# Build engine file
./yolov5 -s ../yolov5n.wts yolov5n.engine n

# Copy engine về thư mục models
cp yolov5n.engine /path/to/JetsonYolo_Project/models/
```

**Cách 2: Build từ ONNX file**

```bash
# Export YOLOv5 sang ONNX (trên máy có GPU hoặc CPU)
cd ~/yolov5
python export.py --weights models/yolov5n.pt --include onnx --imgsz 640

# Trên Jetson Nano: Build engine từ ONNX
cd /path/to/JetsonYolo_Project/scripts
python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine
```

**Cách 3: Sử dụng trtexec (có sẵn trong TensorRT)**

```bash
/usr/src/tensorrt/bin/trtexec --onnx=yolov5n.onnx \
    --saveEngine=yolov5n.engine \
    --fp16 \
    --workspace=4096
```

**Lưu ý quan trọng:**
- Engine phải được build trên chính Jetson Nano (không copy từ máy khác)
- Sử dụng `--fp16` để tối ưu hiệu suất
- Giảm `--workspace` nếu gặp lỗi out of memory

### 3. Build project

```bash
cd JetsonYolo_Project
mkdir -p build
cd build
cmake ..
make -j4
```

File thực thi sẽ được tạo tại `build/JetsonYolo_Project`.

## 🚀 Sử dụng

### Chế độ 1: Object Detection (Mặc định)

Chỉ phát hiện và vẽ bounding box cho các đối tượng:

```bash
./JetsonYolo_Project ../models/yolov5n.engine ../data/test_video.mp4
```

**Tính năng:**
- Phát hiện tất cả 80 classes từ COCO dataset
- Vẽ bounding box và label
- Hiển thị FPS, số lượng detections
- Không có tracking và đếm người

### Chế độ 2: Object Detection + Đếm người vào/ra

Bật tính năng đếm người với tracking:

```bash
./JetsonYolo_Project ../models/yolov5n.engine ../data/test_video.mp4 --count
# Hoặc dùng flag ngắn
./JetsonYolo_Project ../models/yolov5n.engine ../data/test_video.mp4 -c
```

**Tính năng:**
- Tất cả tính năng của chế độ 1
- **Tracking người** qua các frame
- **Đếm người vào/ra** với đường đếm có thể điều chỉnh
- **Visualization**: Hiển thị đường đếm, tracks, và thống kê

**Tham số:**
- `engine_path`: Đường dẫn đến file .engine (bắt buộc)
- `video_path`: Đường dẫn đến video input (bắt buộc)
- `--count` hoặc `-c`: Bật tính năng đếm người (tùy chọn)

**Điều khiển bằng bàn phím:**
- `q` hoặc `ESC`: Thoát chương trình
- **Chỉ khi bật `--count`:**
  - `r` hoặc `R`: Reset counter về 0
  - `l` hoặc `L`: Di chuyển đường đếm sang trái (20 pixels)
  - `k` hoặc `K`: Di chuyển đường đếm sang phải (20 pixels)

**Hiển thị trên video (khi bật `--count`):**
- **Đường đếm màu đỏ**: Đường dọc để đếm người vào/ra (mặc định ở giữa)
- **Điểm màu xanh**: Vị trí center của mỗi người
- **ID màu xanh**: Số ID của mỗi người
- **Đường trail màu vàng**: Hướng di chuyển
- **Thông tin đếm** (góc trên bên trái):
  - `Vao: X`: Số người đi vào (màu xanh lá)
  - `Ra: Y`: Số người đi ra (màu cam)
  - `Tong: Z`: Số người hiện tại (màu vàng)

### So sánh với Python version (optional)

```bash
cd scripts
python compare_fps.py ../models/yolov5n.pt ../data/test_video.mp4
```

## ✨ Tính năng

### 1. Object Detection với YOLOv5
- Phát hiện đối tượng real-time với YOLOv5
- Hỗ trợ 80 classes từ COCO dataset
- Tối ưu hóa với TensorRT cho hiệu suất cao
- Vẽ bounding box và label cho tất cả đối tượng

### 2. Đếm người vào/ra (People Counting)
- **Tracking**: Theo dõi người qua các frame bằng distance-based matching
- **Counting Line**: Đường đếm dọc có thể điều chỉnh
- **Direction Detection**: Tự động phân biệt người đi vào và đi ra
- **Visualization**: Hiển thị trực quan với đường đếm, tracks, và thống kê
- **Real-time**: Cập nhật số đếm theo thời gian thực
- **Hiệu suất**: Chỉ khởi tạo khi được bật, không ảnh hưởng khi tắt

## 🐛 Troubleshooting

### Lỗi: Cannot find TensorRT
- Kiểm tra đường dẫn TensorRT trong `CMakeLists.txt`
- Đảm bảo TensorRT được cài đặt: `dpkg -l | grep tensorrt`

### Lỗi: CUDA out of memory
- Giảm input resolution trong code
- Sử dụng model nhỏ hơn (yolov5n thay vì yolov5s)
- Đóng các ứng dụng khác đang chạy

### Lỗi: Cannot open engine file
- Đảm bảo file .engine đã được build đúng
- Kiểm tra đường dẫn file
- **Quan trọng**: Engine phải được build trên cùng Jetson Nano

### Lỗi khi build engine: Out of memory
- Giảm workspace size: `--workspace 2048` hoặc `--workspace 1024`
- Đóng các ứng dụng khác
- Sử dụng model nhỏ hơn

### Lỗi khi build engine: Unsupported ONNX ops
- Đảm bảo sử dụng YOLOv5 v6.1+ hoặc v7.0+
- Kiểm tra TensorRT version: `dpkg -l | grep tensorrt`
- Có thể cần update TensorRT

### Lỗi: TensorRT version mismatch
- Engine được build với TensorRT version cụ thể
- Nếu update TensorRT, cần rebuild engine
- Kiểm tra version: `python -c "import tensorrt as trt; print(trt.__version__)"`

## 📝 Ghi chú

- File `.engine` phải được build trên cùng một GPU architecture (Jetson Nano)
- Model được train trên COCO dataset (80 classes)
- Output video được lưu với tên `result.avi` (codec MJPG, resize 50%)
- **Tính năng đếm người** (khi bật với `--count`):
  - Chỉ hoạt động với class "person" (class_id = 0)
  - Mỗi người chỉ được đếm một lần khi vượt qua đường đếm
  - Tracks tự động xóa sau 10 frame nếu không phát hiện
  - Đường đếm có thể điều chỉnh bằng phím `l`/`k`
- **Chế độ detection thuần túy** (không có `--count`):
  - Chỉ phát hiện và vẽ bounding box, không có tracking
  - Hiệu suất cao hơn do không có overhead

## 📄 License

Dự án này sử dụng YOLOv5 từ Ultralytics (GPL-3.0 license).

## 👤 Tác giả

Vu Van An - Dau Duc Giap

## 📚 Tài liệu tham khảo

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [TensorRT](https://developer.nvidia.com/tensorrt)
- [Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
