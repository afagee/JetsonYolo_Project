# Jetson Nano - Realtime YOLOv5 Person Counter & Tracker 🚀

Dự án nhận diện và đếm người (Person Counting) được tối ưu hóa chuyên biệt cho **NVIDIA Jetson Nano**.
Hệ thống sử dụng **YOLOv5 v6.1** (TensorRT Engine + FP16), thuật toán Tracker Euclidean cải tiến và xử lý Đa luồng (Multithreading) để đạt hiệu năng **20-30 FPS**.

---

## 📋 Yêu cầu phần cứng

* **Thiết bị:** NVIDIA Jetson Nano (Khuyến nghị bản 4GB Developer Kit).
* **Thẻ nhớ:** MicroSD 32GB trở lên (Class 10 U3).
* **Camera:** Webcam USB (Logitech C270/C920) hoặc Camera CSI (IMX219).
* **Hệ điều hành:** JetPack 4.6.1 (Python 3.6.9).

---

## 🛠️ Hướng dẫn Cài đặt (Step-by-Step)

### Bước 1: Cài đặt thư viện hệ thống
Mở Terminal và chạy các lệnh sau để cài đặt các gói phụ thuộc cần thiết:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev libomp-dev
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
```
### Bước 2: Cài đặt PyTorch & Torchvision (QUAN TRỌNG)
⚠️ LƯU Ý: KHÔNG dùng lệnh ```pip install torch```. Bạn phải cài bản hỗ trợ GPU (aarch64) từ NVIDIA.

1. Cài đặt PyTorch v1.10.0 (Cho JetPack 4.6):

```Bash
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```
2. Cài đặt Torchvision v0.11.1:

```Bash
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user
cd ..
```
Kiểm tra: Chạy ```python3 -c "import torch; print(torch.cuda.is_available())"```. Nếu hiện ```True``` là thành công.
### Bước 3: Cài đặt Project & YOLOv5 v6.1
Copy code dự án (main.py, tracker.py, config.yaml...) vào thư mục làm việc (ví dụ: JetsonCounter).

1. Clone YOLOv5 version 6.1: Bắt buộc dùng bản này để tương thích tốt nhất với Python 3.6 trên Nano.

```Bash
# Clone đúng phiên bản v6.1
git clone --branch v6.1 https://github.com/ultralytics/yolov5

# Chỉnh sửa requirements của YOLOv5 để tránh xung đột với PyTorch đã cài
cd yolov5
sed -i 's/torch>=.*/# torch/g' requirements.txt
sed -i 's/torchvision>=.*/# torchvision/g' requirements.txt

# Cài đặt thư viện phụ cho YOLOv5
pip3 install -r requirements.txt
cd ..
```
2. Cài đặt các thư viện phụ của dự án:

```Bash
pip3 install numpy>=1.18.5 opencv-python>=4.1.1 PyYAML>=5.3.1 psutil tqdm
```
⚡ Tối ưu hóa Model (TensorRT)
Để đạt FPS cao, BẮT BUỘC phải chuyển đổi model .pt sang .engine ngay trên Jetson Nano.

1. Tải Model Weights (Phiên bản v6.1):

```Bash
cd yolov5

# Tải YOLOv5s (Small) - Khuyên dùng (Chính xác & Nhanh vừa phải)
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt

# Hoặc Tải YOLOv5n (Nano) - Nếu cần tốc độ cực nhanh (>30 FPS)
# wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt
```
2. Convert sang Engine (Mất khoảng 15 phút): Chạy lệnh export ngay trên Nano:

```Bash
# Dùng yolov5s (Small) - img size 512
python3 export.py --weights yolov5s.pt --include engine --img 512 --device 0 --half

# Hoặc dùng yolov5n (Nano) - img size 416
# python3 export.py --weights yolov5n.pt --include engine --img 416 --device 0 --half
```
## ⚙️ Cấu hình (config.yaml)
Tạo hoặc sửa file config.yaml tại thư mục gốc dự án:

```YAML
# --- MODEL ---
weights: "yolov5s.engine"   # File engine vừa tạo (yolov5s.engine hoặc yolov5n.engine)
img_size: 512               # Phải KHỚP với lệnh export ở trên (512 hoặc 416)
device: "0"                 # 0 là GPU
conf_thres: 0.4             # Độ tin cậy (Cao = ít nhiễu)
classes: [0]                # 0: Person

# --- TRACKER ---
dist_threshold: 150         # Khoảng cách tối đa nối ID (pixel)
max_disappeared: 5          # Số frame chờ trước khi xóa ID (giúp ID ổn định)

# --- HIỂN THỊ ---
source: "test.mp4"          # File video hoặc "0" (Webcam)
process_width: 960          # Resize ảnh để xử lý nhẹ hơn (Khuyên dùng 960x540)
process_height: 540
line_coords: [480, 0, 480, 540] # [x1, y1, x2, y2] - Đường kẻ đếm
```
## ▶️ Chạy chương trình
1. Kích hoạt chế độ hiệu năng cao (Bắt buộc): Chạy lệnh này mỗi khi khởi động lại Nano để quạt quay mạnh hơn và CPU/GPU chạy max xung.
```Bash
sudo jetson_clocks
```
2. Chạy ứng dụng:

```Bash
python3 main.py
```
## ❓ Xử lý lỗi thường gặp
1. Lỗi AttributeError: 'NoneType' object has no attribute 'create_execution_context'

- Nguyên nhân: File .engine bị lỗi hoặc bạn copy file engine từ máy tính khác (PC) sang Nano.

- Khắc phục: Xóa file .engine cũ đi và chạy lại bước "Tối ưu hóa Model" ngay trên chính Jetson Nano.

2. Lỗi SystemError: initialization of _internal failed without raising an exception

- Nguyên nhân: Xung đột phiên bản numpy.

- Khắc phục:

```Bash
pip3 install numpy==1.19.4
```
3. Video lag, FPS thấp (< 10 FPS)

- Kiểm tra xem đã chạy sudo jetson_clocks chưa.

- Kiểm tra config.yaml xem weights đang là .pt hay .engine. Phải dùng .engine mới nhanh.