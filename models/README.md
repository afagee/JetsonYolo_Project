# Models Directory

Thư mục này chứa các file model cho YOLOv5:

- `yolov5n.pt`: Model PyTorch gốc (download từ ultralytics)
- `yolov5n.wts`: File trung gian (được convert từ .pt)
- `yolov5n.engine`: TensorRT engine đã build cho Jetson Nano

**Lưu ý**: Các file này thường rất lớn và không được commit lên Git (đã được thêm vào .gitignore).

## Download model

```bash
# Download YOLOv5n model
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
```

## Build TensorRT engine

Sử dụng yolov5 repository hoặc TensorRT sample để build engine từ .pt hoặc .wts file.

