"""
Script Python để chạy YOLOv5 và so sánh FPS với phiên bản C++ TensorRT
Script này chạy chậm hơn để làm baseline so sánh
"""

import cv2
import torch
import time
import sys
from pathlib import Path

# Thêm yolov5 vào path (cần clone yolov5 repository)
# sys.path.append('/path/to/yolov5')

def run_yolov5_python(model_path, video_path, conf_threshold=0.5):
    """
    Chạy YOLOv5 bằng Python (PyTorch) để benchmark
    
    Args:
        model_path: Đường dẫn đến model .pt
        video_path: Đường dẫn đến video
        conf_threshold: Ngưỡng confidence
    """
    print(f"Loading model from: {model_path}")
    
    # Load model (cần yolov5 repository)
    try:
        # Cách 1: Sử dụng yolov5 package
        import yolov5
        model = yolov5.load(model_path)
        model.conf = conf_threshold
    except ImportError:
        # Cách 2: Sử dụng torch.hub
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model.conf = conf_threshold
    
    print("Model loaded successfully!")
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    # Thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height} @ {fps} FPS")
    
    # Benchmark
    frame_count = 0
    total_time = 0.0
    
    print("\nStarting inference...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inference
        start_time = time.time()
        results = model(frame)
        inference_time = time.time() - start_time
        
        total_time += inference_time
        frame_count += 1
        
        # Tính FPS
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        
        # Vẽ kết quả
        annotated_frame = results.render()[0]
        
        # Hiển thị FPS
        cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Detections: {len(results.xyxy[0])}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv5 Python", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kết quả
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print("\n=== Python Benchmark Results ===")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print("\nNote: C++ TensorRT version should be significantly faster!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_fps.py <model.pt> <video.mp4> [conf_threshold]")
        print("Example: python compare_fps.py ../models/yolov5n.pt ../data/test_video.mp4 0.5")
        sys.exit(1)
    
    model_path = sys.argv[1]
    video_path = sys.argv[2]
    conf_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    run_yolov5_python(model_path, video_path, conf_threshold)

