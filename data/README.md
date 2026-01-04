# Data Directory

Thư mục này chứa dữ liệu test cho dự án, chủ yếu là các file video để chạy inference.

## 📁 Nội dung

- `test_video.mp4`: Video test mẫu để chạy inference (nếu có)

## 📝 Lưu ý

- Các file video thường rất lớn và **không được commit lên Git** (đã được thêm vào `.gitignore`)
- Bạn có thể đặt video của mình vào thư mục này để test

## 🚀 Sử dụng

### Sử dụng video của bạn

1. Đặt video test vào thư mục này:
   ```bash
   cp /path/to/your/video.mp4 data/
   ```

2. Chạy inference:
   ```bash
   cd build
   ./JetsonYolo_Project ../models/yolov5n.engine ../data/your_video.mp4
   ```

3. Với tính năng đếm người:
   ```bash
   ./JetsonYolo_Project ../models/yolov5n.engine ../data/your_video.mp4 --count
   ```

### Định dạng video hỗ trợ

- MP4 (khuyến nghị)
- AVI
- MOV
- Các định dạng khác được OpenCV hỗ trợ

### Yêu cầu video

- **Độ phân giải**: Không giới hạn, nhưng video lớn sẽ chạy chậm hơn
- **FPS**: Tùy ý, chương trình sẽ xử lý theo FPS của video
- **Codec**: H.264, H.265, hoặc các codec được OpenCV hỗ trợ

## 📊 Output

Video kết quả sẽ được lưu với tên `result.avi` trong thư mục gốc của project (nơi chạy chương trình), không phải trong thư mục `data/`.

Video output sẽ được resize xuống 50% kích thước gốc để tối ưu hiệu suất ghi file.
