# JetsonYolo_Project - YOLOv5 TensorRT trên Jetson Nano

Dự án triển khai YOLOv5 với TensorRT trên Jetson Nano để tối ưu hóa hiệu suất inference.

## 📁 Cấu trúc thư mục

```
JetsonYolo_Project/
├── CMakeLists.txt          # File cấu hình build
├── README.md               # Hướng dẫn sử dụng
├── .gitignore              # Loại bỏ file rác khi up lên Git
├── build/                  # Nơi chứa file thực thi sau khi biên dịch
├── data/                   # Chứa dữ liệu test
│   └── test_video.mp4
├── include/                # Chứa các file Header (.h)
│   ├── common.hpp          # Các struct chung (Detection, Box...)
│   ├── cuda_utils.h        # Hàm kiểm tra lỗi CUDA
│   ├── logging.h           # Logger bắt buộc của TensorRT
│   └── yolov5.hpp          # Khai báo Class YOLOv5
├── models/                 # Chứa weights
│   ├── yolov5n.pt          # Model gốc (để tham khảo)
│   ├── yolov5n.wts         # File trung gian
│   └── yolov5n.engine      # File engine đã build cho Nano
├── scripts/                # Các file Python phụ trợ
│   ├── gen_wts.py          # Script convert .pt -> .wts
│   ├── build_engine.py     # Script build TensorRT engine từ ONNX
│   └── compare_fps.py      # Code Python chạy chậm (để so sánh benchmark)
└── src/                    # Chứa mã nguồn C++ (.cpp)
    ├── main.cpp            # Hàm main: Đọc video, gọi YOLO, tính FPS
    └── yolov5.cpp          # Cài đặt chi tiết các hàm của Class YOLOv5
```

## 🔧 Yêu cầu hệ thống

- **Hardware**: NVIDIA Jetson Nano
- **OS**: JetPack 4.6+ hoặc JetPack 5.x
- **CUDA**: 10.2+ (với JetPack 4.6) hoặc 11.4+ (với JetPack 5.x)
- **TensorRT**: 8.0+
- **OpenCV**: 4.5+
- **CMake**: 3.10+

## 📦 Cài đặt dependencies

### 1. Cài đặt OpenCV (nếu chưa có)

```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

### 2. Kiểm tra TensorRT

```bash
dpkg -l | grep tensorrt
```

TensorRT thường được cài đặt sẵn với JetPack.

## 🚀 Hướng dẫn sử dụng

### Bước 1: Chuẩn bị model

1. **Download YOLOv5 model** (.pt file):
   ```bash
   # Tải từ ultralytics hoặc sử dụng model có sẵn
   wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
   mv yolov5n.pt models/
   ```

2. **Convert .pt sang .wts** (nếu cần):
   ```bash
   cd scripts
   python gen_wts.py ../models/yolov5n.pt ../models/yolov5n.wts
   ```

3. **Build TensorRT engine trên Jetson Nano**:

   Có 3 cách để build TensorRT engine:

   **Cách 1: Build từ file .wts (Khuyến nghị cho Jetson Nano)**

   ```bash
   # Clone yolov5 repository (nếu chưa có)
   cd ~
   git clone https://github.com/wang-xinyu/tensorrtx.git
   cd tensorrtx/yolov5
   
   # Copy file .wts vào thư mục này
   cp /path/to/JetsonYolo_Project/models/yolov5n.wts .
   
   # Build engine (cho Jetson Nano)
   # Lưu ý: Jetson Nano có GPU compute capability 5.3
   mkdir build
   cd build
   cmake ..
   make
   
   # Chạy để build engine (input size mặc định 640x640)
   ./yolov5 -s ../yolov5n.wts yolov5n.engine n
   
   # Hoặc với input size khác (ví dụ 416x416)
   ./yolov5 -s ../yolov5n.wts yolov5n_416.engine n 416
   
   # Copy engine về thư mục models
   cp yolov5n.engine /path/to/JetsonYolo_Project/models/
   ```

   **Cách 2: Build từ file .pt trực tiếp (YOLOv5 v7.0+)**

   ```bash
   # Clone yolov5 repository
   cd ~
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   
   # Cài đặt dependencies
   pip install -r requirements.txt
   
   # Export sang ONNX (trên máy có GPU hoặc CPU)
   python export.py --weights models/yolov5n.pt --include onnx --imgsz 640
   
   # Copy file .onnx lên Jetson Nano (nếu export trên máy khác)
   # scp yolov5n.onnx user@jetson-nano:/path/to/models/
   
   # Trên Jetson Nano: Convert ONNX sang TensorRT engine
   # Sử dụng trtexec (có sẵn trong TensorRT)
   /usr/src/tensorrt/bin/trtexec --onnx=yolov5n.onnx \
       --saveEngine=yolov5n.engine \
       --fp16 \
       --workspace=4096 \
       --verbose
   
   # Hoặc sử dụng Python API
   python -c "
   import tensorrt as trt
   
   TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
   builder = trt.Builder(TRT_LOGGER)
   network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
   parser = trt.OnnxParser(network, TRT_LOGGER)
   
   with open('yolov5n.onnx', 'rb') as model:
       parser.parse(model.read())
   
   config = builder.create_builder_config()
   config.max_workspace_size = 1 << 30  # 1GB
   config.set_flag(trt.BuilderFlag.FP16)
   
   engine = builder.build_engine(network, config)
   with open('yolov5n.engine', 'wb') as f:
       f.write(engine.serialize())
   "
   
   # Copy engine về thư mục models
   cp yolov5n.engine /path/to/JetsonYolo_Project/models/
   ```

   **Cách 3: Sử dụng script Python tự động (Dễ nhất)**

   ```bash
   # Export YOLOv5 sang ONNX (nếu chưa có)
   # Trên máy có GPU hoặc trên Jetson Nano:
   python export.py --weights models/yolov5n.pt --include onnx --imgsz 640
   
   # Trên Jetson Nano: Build engine từ ONNX
   cd scripts
   python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine
   
   # Với các tùy chọn:
   # - Sử dụng FP32 thay vì FP16:
   python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine --fp32
   
   # - Giảm workspace size nếu gặp lỗi memory:
   python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine --workspace 2048
   ```

   **Lưu ý cho Jetson Nano (JetPack 4.6):**

   - TensorRT 8.0+ yêu cầu JetPack 4.6 trở lên
   - Nếu dùng JetPack 4.5 hoặc cũ hơn, cần dùng TensorRT 7.x
   - Đảm bảo build engine trên chính Jetson Nano (không build trên máy khác rồi copy sang)
   - Engine được build cho một GPU architecture cụ thể, không thể dùng chéo giữa các GPU khác nhau
   - Sử dụng `--fp16` để tối ưu hiệu suất trên Jetson Nano
   - Giảm `--workspace` nếu gặp lỗi out of memory (mặc định 4096MB)

### Bước 2: Build project

```bash
cd JetsonYolo_Project
mkdir -p build
cd build
cmake ..
make -j4
```

### Bước 3: Chạy inference

```bash
# Chạy với video
./JetsonYolo_Project ../models/yolov5n.engine ../data/test_video.mp4

# Với custom thresholds
./JetsonYolo_Project ../models/yolov5n.engine ../data/test_video.mp4 0.5 0.4
```

**Tham số:**
- `engine_path`: Đường dẫn đến file .engine
- `video_path`: Đường dẫn đến video input
- `conf_threshold`: Ngưỡng confidence (mặc định: 0.5)
- `nms_threshold`: Ngưỡng NMS (mặc định: 0.4)

### Bước 4: So sánh với Python (optional)

```bash
cd scripts
python compare_fps.py ../models/yolov5n.pt ../data/test_video.mp4
```

## 📊 Benchmark

Dự án này được tối ưu để đạt hiệu suất cao trên Jetson Nano:
- **FPS**: Tùy thuộc vào model size và video resolution
- **Latency**: Thấp hơn đáng kể so với PyTorch version
- **Memory**: Tối ưu hóa bộ nhớ với TensorRT

## 🐛 Troubleshooting

### Lỗi: Cannot find TensorRT
- Kiểm tra đường dẫn TensorRT trong `CMakeLists.txt`
- Đảm bảo TensorRT được cài đặt đúng

### Lỗi: CUDA out of memory
- Giảm input resolution trong code
- Sử dụng model nhỏ hơn (yolov5n thay vì yolov5s)

### Lỗi: Cannot open engine file
- Đảm bảo file .engine đã được build đúng
- Kiểm tra đường dẫn file
- Đảm bảo engine được build trên cùng Jetson Nano (không copy từ máy khác)

### Lỗi khi build engine: Out of memory
- Giảm workspace size: `--workspace 2048` hoặc `--workspace 1024`
- Đóng các ứng dụng khác đang chạy
- Sử dụng model nhỏ hơn (yolov5n thay vì yolov5s)

### Lỗi khi build engine: Unsupported ONNX ops
- Đảm bảo sử dụng YOLOv5 v7.0+ (hỗ trợ export ONNX tốt hơn)
- Kiểm tra TensorRT version: `dpkg -l | grep tensorrt`
- Có thể cần update TensorRT lên phiên bản mới hơn

### Lỗi: TensorRT version mismatch
- Engine được build với TensorRT version cụ thể
- Nếu update TensorRT, cần rebuild engine
- Kiểm tra version: `python -c "import tensorrt as trt; print(trt.__version__)"`

## 📝 Ghi chú

- File `.engine` phải được build trên cùng một GPU architecture (Jetson Nano)
- Model được train trên COCO dataset (80 classes)
- Output video sẽ được lưu với tên `output_result.mp4`

## 📄 License

Dự án này sử dụng YOLOv5 từ Ultralytics (GPL-3.0 license).

## 👤 Tác giả

Dự án được tạo cho mục đích học tập và nghiên cứu.

