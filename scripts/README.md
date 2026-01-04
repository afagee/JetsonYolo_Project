# Scripts Directory

Thư mục này chứa các script Python phụ trợ để hỗ trợ quá trình build model và benchmark.

## 📁 Nội dung

- `gen_wts.py`: Convert file .pt (PyTorch) sang .wts (định dạng cho tensorrtx)
- `build_engine.py`: Build TensorRT engine từ file ONNX
- `compare_fps.py`: So sánh FPS giữa TensorRT version (C++) và PyTorch version (Python)

## 🔧 Yêu cầu

- Python 3.6+
- PyTorch (cho `gen_wts.py` và `compare_fps.py`)
- TensorRT Python API (cho `build_engine.py`)
- OpenCV Python (cho `compare_fps.py`)
- YOLOv5 repository (cho `gen_wts.py`)

## 📝 Chi tiết từng script

### 1. gen_wts.py

**Mục đích**: Convert file model PyTorch (.pt) sang định dạng .wts để sử dụng với tensorrtx.

**Cách sử dụng**:
```bash
cd scripts
python gen_wts.py ../models/yolov5n.pt ../models/yolov5n.wts
```

**Tham số**:
- `-w, --weights`: Đường dẫn đến file .pt (bắt buộc)
- `-o, --output`: Đường dẫn đến file .wts output (tùy chọn, mặc định là cùng tên với .pt)
- `-t, --type`: Loại model - `detect`, `cls`, hoặc `seg` (mặc định: `detect`)

**Lưu ý**:
- Script này cần import từ YOLOv5 repository (`utils.torch_utils`)
- Đảm bảo đã clone YOLOv5 repository và cài đặt dependencies
- Script sẽ load model và extract weights vào định dạng .wts

### 2. build_engine.py

**Mục đích**: Build TensorRT engine từ file ONNX sử dụng TensorRT Python API.

**Cách sử dụng**:
```bash
cd scripts
python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine
```

**Tham số**:
- `input.onnx`: Đường dẫn đến file ONNX (bắt buộc)
- `output.engine`: Đường dẫn đến file engine output (bắt buộc)
- `--fp32`: Sử dụng FP32 thay vì FP16 (tùy chọn)
- `--workspace SIZE_MB`: Kích thước workspace (MB, mặc định: 4096)

**Ví dụ**:
```bash
# Build với FP16 (mặc định, khuyến nghị)
python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine

# Build với FP32
python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine --fp32

# Build với workspace nhỏ hơn (nếu gặp lỗi memory)
python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine --workspace 2048
```

**Lưu ý**:
- Script này sử dụng TensorRT Python API
- Quá trình build có thể mất vài phút
- Nếu gặp lỗi out of memory, giảm workspace size
- FP16 được khuyến nghị cho Jetson Nano để tối ưu hiệu suất

### 3. compare_fps.py

**Mục đích**: So sánh hiệu suất (FPS) giữa TensorRT version (C++) và PyTorch version (Python).

**Cách sử dụng**:
```bash
cd scripts
python compare_fps.py ../models/yolov5n.pt ../data/test_video.mp4
```

**Tham số**:
- `model_path`: Đường dẫn đến file .pt (PyTorch model)
- `video_path`: Đường dẫn đến video test

**Chức năng**:
- Load model PyTorch
- Chạy inference trên video
- Tính và hiển thị FPS trung bình
- So sánh với kết quả từ TensorRT version (C++)

**Lưu ý**:
- Script này chạy chậm hơn nhiều so với TensorRT version
- Chủ yếu dùng để benchmark và so sánh
- Cần cài đặt PyTorch và YOLOv5

## 🚀 Workflow khuyến nghị

### Workflow 1: Sử dụng tensorrtx (từ .wts)

```bash
# 1. Download model
cd models
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt

# 2. Convert .pt -> .wts
cd ../scripts
python gen_wts.py ../models/yolov5n.pt ../models/yolov5n.wts

# 3. Build engine bằng tensorrtx (xem README chính)
# ... sử dụng tensorrtx repository ...
```

### Workflow 2: Sử dụng ONNX (từ .pt)

```bash
# 1. Download model
cd models
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt

# 2. Export .pt -> .onnx (sử dụng YOLOv5 export.py)
cd ~/yolov5
python export.py --weights /path/to/models/yolov5n.pt --include onnx --imgsz 640

# 3. Build engine từ ONNX
cd /path/to/JetsonYolo_Project/scripts
python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine
```

## ⚠️ Troubleshooting

### Lỗi: ModuleNotFoundError khi chạy gen_wts.py
- Đảm bảo đã clone YOLOv5 repository
- Cài đặt dependencies: `pip install -r requirements.txt` (trong YOLOv5 repo)
- Chạy script từ thư mục có thể import YOLOv5 utils

### Lỗi: TensorRT not found khi chạy build_engine.py
- Kiểm tra TensorRT đã được cài đặt: `dpkg -l | grep tensorrt`
- Kiểm tra Python có thể import: `python -c "import tensorrt as trt; print(trt.__version__)"`

### Lỗi: Out of memory khi build engine
- Giảm workspace size: `--workspace 2048` hoặc `--workspace 1024`
- Đóng các ứng dụng khác đang chạy
- Sử dụng model nhỏ hơn (yolov5n thay vì yolov5s)

