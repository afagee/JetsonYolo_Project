# main.py
import sys
import time
import cv2
import torch
import numpy as np
import psutil
import os
import yaml
from pathlib import Path
from threading import Thread

# Thêm đường dẫn YOLOv5
sys.path.append(str(Path(__file__).parent / 'yolov5'))

try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes, check_img_size
    from utils.augmentations import letterbox
    from utils.torch_utils import select_device
except ImportError:
    print("LỖI: Không tìm thấy folder 'yolov5'.")
    sys.exit()

# IMPORT SORT THAY VÌ TRACKER CŨ
from sort import Sort

# --- HÀM TẠO PIPELINE GSTREAMER (QUAN TRỌNG CHO JETSON) ---
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=960,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

class CameraLoader:
    def __init__(self, src=0, use_gstreamer=False):
        if use_gstreamer and isinstance(src, int):
            # Nếu dùng Camera CSI (Ribbon cable)
            gstreamer_str = gstreamer_pipeline(sensor_id=src)
            self.stream = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
            print(f"--- Đang dùng GStreamer Pipeline: {gstreamer_str} ---")
        else:
            # Nếu dùng USB Camera hoặc File Video
            self.stream = cv2.VideoCapture(src)
            
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()
            # Ngủ cực ngắn để không chiếm CPU của luồng chính
            time.sleep(0.005) 

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True

def load_config(config_path='config.yaml'):
    if not os.path.exists(config_path): config_path = 'yolov5/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)

def get_jetson_stats():
    cpu = psutil.cpu_percent()
    gpu = 0
    try:
        # Đọc tải GPU TeX (Tegra)
        with open("/sys/devices/gpu.0/load", "r") as f:
            gpu = int(f.read().strip()) / 10.0
    except: pass
    return cpu, gpu

# --- LOGIC ĐẾM NGƯỜI ---
def intersect(A, B, C, D):
    def ccw(p1, p2, p3):
        return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def vector_cross_product(line_start, line_end, point):
    v_line = np.array([line_end[1] - line_start[1], line_end[0] - line_start[0]])
    v_point = np.array([point[1] - line_start[1], point[0] - line_start[0]])
    return np.cross(v_line, v_point)

def main():
    cfg = load_config()
    print(f"--- POWER MODE: {cfg['weights']} | Size: {cfg['img_size']} ---")

    # 1. Setup Model
    device = select_device(cfg['device'])
    model = DetectMultiBackend(cfg['weights'], device=device, dnn=False, fp16=True) # Bật FP16
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(cfg['img_size'], s=stride)
    
    # 2. Setup Camera
    src = cfg['source']
    use_gst = False
    if str(src).isdigit(): 
        src = int(src)
        use_gst = True # Tự động bật GStreamer nếu input là số (Camera)
    
    cam = CameraLoader(src, use_gstreamer=use_gst).start()
    time.sleep(1.0)

    # 3. Setup SORT Tracker (Mới)
    # max_age: số frame giữ ID khi bị mất dấu (quan trọng khi bị che khuất)
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    
    track_history = {}
    count_in, count_out = 0, 0
    line = cfg['line_coords'] 
    
    # Pre-calculate line vector for speed
    line_start = (line[0], line[1])
    line_end = (line[2], line[3])

    window_name = "Jetson Pro SORT"

    prev_time = time.time()
    fps = 0
    frame_idx = 0

    while True:
        grabbed, frame = cam.read()
        if not grabbed:
            if isinstance(src, str): break 
            continue

        frame_idx += 1
        
        # --- PRE-PROCESS ---
        img = letterbox(frame, imgsz, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if model.fp16 else img.float()
        img /= 255.0
        if len(img.shape) == 3: img = img[None]

        # --- INFERENCE ---
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, cfg['conf_thres'], cfg['iou_thres'], classes=cfg['classes'])

        # --- PREPARE FOR TRACKER ---
        dets_to_sort = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                # SORT cần format: [x1, y1, x2, y2, score]
                # Code gốc output: [x1, y1, x2, y2, conf, cls]
                for *xyxy, conf, cls in det:
                    dets_to_sort.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item()])
        
        dets_to_sort = np.array(dets_to_sort)
        if len(dets_to_sort) == 0:
            dets_to_sort = np.empty((0, 5))

        # --- TRACKING UPDATE ---
        tracked_dets = tracker.update(dets_to_sort)
        # tracked_dets trả về format: [cx, cy, w, h, id] (do hàm Sort đã convert lại)

        # --- DRAWING & COUNTING ---
        cv2.line(frame, line_start, line_end, (0, 255, 255), 2)

        for trk in tracked_dets:
            # Sort trả về float, cần ép kiểu int
            cx, cy, w, h, trk_id = trk
            cx, cy, w, h, trk_id = int(cx), int(cy), int(w), int(h), int(trk_id)
            
            # Tính lại toạ độ box để vẽ (vì SORT trả về cx, cy)
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
            cv2.putText(frame, str(trk_id), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            # Logic Đếm (Giữ nguyên logic vector cũ của bạn vì nó tốt)
            curr_point = (cx, cy)
            if trk_id not in track_history:
                track_history[trk_id] = [curr_point]
            else:
                track_history[trk_id].append(curr_point)
                if len(track_history[trk_id]) > 30: track_history[trk_id].pop(0)

            if len(track_history[trk_id]) >= 2:
                prev_point = track_history[trk_id][-2]
                
                # Kiểm tra cắt đường line
                if intersect(line_start, line_end, prev_point, curr_point):
                    cross = vector_cross_product(line_start, line_end, curr_point)
                    
                    # Thêm điều kiện: Phải di chuyển một đoạn đủ xa để tránh nhiễu
                    dist_moved = np.linalg.norm(np.array(curr_point) - np.array(prev_point))
                    
                    if dist_moved > 5: # Chỉ đếm nếu di chuyển > 5 pixels
                        if cross > 0: 
                            count_in += 1
                            cv2.line(frame, line_start, line_end, (0, 0, 255), 4) # Hiệu ứng nháy đỏ
                        else: 
                            count_out += 1
                            cv2.line(frame, line_start, line_end, (255, 0, 255), 4) # Hiệu ứng nháy tím
                        
                        # Xóa lịch sử để tránh đếm trùng ngay lập tức
                        track_history[trk_id] = []

        # --- STATS ---
        curr_time = time.time()
        fps = (fps * 0.9) + (1/(curr_time - prev_time) * 0.1)
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"IN: {count_in} OUT: {count_out}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == ord('q'): break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()