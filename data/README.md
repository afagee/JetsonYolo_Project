# Data Directory

Thư mục này chứa dữ liệu test cho dự án:

- `test_video.mp4`: Video test để chạy inference

**Lưu ý**: File video thường rất lớn và không được commit lên Git (đã được thêm vào .gitignore).

## Sử dụng video của bạn

Đặt video test vào thư mục này và chạy:

```bash
./JetsonYolo_Project ../models/yolov5n.engine ../data/your_video.mp4
```

