import cv2
import torch
import time
import sys
import threading
from queue import Queue
from pathlib import Path

# --- CLASS GHI VIDEO ĐA LUỒNG ---
class VideoWriterThread:
    def __init__(self, path, fps, width, height, queue_size=30):
        self.path = path
        # Dùng codec mp4v (software) hoặc thử dùng 'avc1' nếu có hỗ trợ
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(path, self.fourcc, fps, (width, height))
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        
        # Khởi động thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def write(self, frame):
        if self.stopped:
            return
        # Nếu queue đầy, ta bỏ qua frame này để không làm chậm luồng chính
        if not self.queue.full():
            self.queue.put(frame)

    def update(self):
        while True:
            if self.stopped and self.queue.empty():
                break
            
            # Lấy frame từ queue để ghi
            try:
                # timeout=1 để thread có thể check cờ stopped
                frame = self.queue.get(timeout=1)
                self.writer.write(frame)
                self.queue.task_done()
            except:
                continue
        
        self.writer.release()

    def stop(self):
        self.stopped = True
        # Đợi thread ghi nốt các frame còn lại trong queue
        self.thread.join()
# ------------------------------

def run_yolov5_python(model_path, video_path, conf_threshold=0.4):
    print(f"Loading model from: {model_path}")
    
    # Load model
    try:
        import yolov5
        model = yolov5.load(model_path)
    except ImportError:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    
    model.conf = conf_threshold
    print("Model loaded successfully!")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        return
    
    # Thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- CẤU HÌNH LƯU VIDEO (THREAD) ---
    input_path_obj = Path(video_path)
    output_filename = input_path_obj.stem + "_output" + input_path_obj.suffix
    output_path = str(input_path_obj.parent / output_filename)
    
    print(f"Saving output video to: {output_path}")
    
    # Khởi tạo Thread ghi video
    video_writer = VideoWriterThread(output_path, fps, width, height)
    # -----------------------------------
    
    frame_count = 0
    total_time = 0.0
    
    print("\nStarting inference...")
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
        annotated_frame = results.render()[0].copy()
        
        cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # --- ĐẨY VÀO THREAD GHI VIDEO ---
        # Hàm này sẽ trả về ngay lập tức, không chờ ghi đĩa
        video_writer.write(annotated_frame)
        # -------------------------------

        cv2.imshow("YOLOv5 Python", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kết thúc
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nAverage FPS: {avg_fps:.2f}")
    
    # Dừng thread và giải phóng
    video_writer.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_fps.py <model.pt> <video.mp4>")
        sys.exit(1)
    
    run_yolov5_python(sys.argv[1], sys.argv[2])
