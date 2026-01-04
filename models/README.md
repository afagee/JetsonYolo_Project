# Models Directory

Thư mục này chứa các file model cho YOLOv5, bao gồm model gốc, file trung gian, và TensorRT engine.

## 📁 Nội dung

- `yolov5n.pt`: Model PyTorch gốc (download từ ultralytics)
- `yolov5n.wts`: File trung gian (được convert từ .pt, dùng cho tensorrtx)
- `yolov5n.engine`: TensorRT engine đã build cho Jetson Nano (file này rất quan trọng)

## 📝 Lưu ý

- Các file model thường rất lớn (từ vài chục MB đến vài trăm MB) và **không được commit lên Git** (đã được thêm vào `.gitignore`)
- File `.engine` phải được build trên chính Jetson Nano mà bạn sẽ chạy inference
- Engine được build cho một GPU architecture cụ thể, không thể dùng chéo giữa các GPU khác nhau

## 🔽 Download model

### Download YOLOv5n model (khuyến nghị cho Jetson Nano)

```bash
cd models
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt
```

Hoặc các model khác:
- `yolov5s.pt`: Small model (lớn hơn, chính xác hơn)
- `yolov5m.pt`: Medium model
- `yolov5l.pt`: Large model
- `yolov5x.pt`: Extra large model

**Lưu ý**: Model càng lớn thì càng chính xác nhưng chạy chậm hơn. Với Jetson Nano, khuyến nghị dùng `yolov5n` (nano).

## 🔄 Convert model

### Convert .pt sang .wts (cho tensorrtx)

```bash
cd scripts
python gen_wts.py ../models/yolov5n.pt ../models/yolov5n.wts
```

**Yêu cầu:**
- Cần có PyTorch và YOLOv5 repository
- Script này cần import từ YOLOv5 utils

### Export .pt sang ONNX (cho TensorRT trực tiếp)

```bash
# Clone YOLOv5 repository
cd ~
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

# Export sang ONNX
python export.py --weights models/yolov5n.pt --include onnx --imgsz 640
```

## 🔨 Build TensorRT engine

Có 3 cách để build TensorRT engine:

### Cách 1: Build từ .wts (Khuyến nghị)

Xem hướng dẫn chi tiết trong `README.md` chính của project.

### Cách 2: Build từ ONNX

```bash
cd scripts
python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine
```

Với các tùy chọn:
```bash
# Sử dụng FP32 thay vì FP16
python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine --fp32

# Giảm workspace size nếu gặp lỗi memory
python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine --workspace 2048
```

### Cách 3: Sử dụng trtexec

```bash
/usr/src/tensorrt/bin/trtexec --onnx=yolov5n.onnx \
    --saveEngine=yolov5n.engine \
    --fp16 \
    --workspace=4096
```

## ⚠️ Lưu ý quan trọng

1. **Engine phải build trên Jetson Nano**: Không thể copy engine từ máy khác sang
2. **TensorRT version**: Engine được build với TensorRT version cụ thể, nếu update TensorRT cần rebuild
3. **GPU architecture**: Engine chỉ chạy trên GPU cùng architecture (Jetson Nano = compute capability 5.3)
4. **FP16 vs FP32**: Sử dụng FP16 để tối ưu hiệu suất trên Jetson Nano
5. **Workspace size**: Nếu gặp lỗi out of memory khi build, giảm workspace size

## 📊 So sánh model sizes

| Model | Size (.pt) | Size (.engine) | FPS (Jetson Nano) | Accuracy |
|-------|------------|----------------|-------------------|----------|
| yolov5n | ~6 MB | ~12 MB | ~30-40 | Thấp nhất |
| yolov5s | ~14 MB | ~28 MB | ~15-25 | Trung bình |
| yolov5m | ~42 MB | ~84 MB | ~5-10 | Khá cao |
| yolov5l | ~90 MB | ~180 MB | ~2-5 | Cao |
| yolov5x | ~170 MB | ~340 MB | ~1-3 | Cao nhất |

**Khuyến nghị cho Jetson Nano**: Sử dụng `yolov5n` để đạt FPS tốt nhất.
