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

from yolov5.tracker import EuclideanDistTracker

# --- CLASS ĐỌC VIDEO ĐA LUỒNG (BÍ QUYẾT TĂNG TỐC) ---
class CameraLoader:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.fps_cap = 0

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed: # Hết video hoặc lỗi
                self.stopped = True
            
            # Giảm tải thread đọc nếu cần (để tránh ngốn CPU quá mức)
            time.sleep(0.01) 

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
        with open("/sys/devices/gpu.0/load", "r") as f:
            gpu = int(f.read().strip()) / 10.0
    except: pass
    return cpu, gpu

# --- HÀM TOÁN HỌC ---
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
    model = DetectMultiBackend(cfg['weights'], device=device, dnn=False, fp16=False)
    stride, pt = model.stride, model.pt
    imgsz = cfg['img_size']
    if pt: imgsz = check_img_size(imgsz, s=stride)
    use_fp16 = (model.fp16 and device.type != 'cpu')

    # 2. Setup Camera Loader (Đa luồng)
    src = cfg['source']
    if str(src).isdigit(): src = int(src)
    
    # Khởi động luồng đọc camera
    cam = CameraLoader(src).start()
    time.sleep(1.0) # Chờ cam khởi động

    # 3. Setup Tracker Nâng cao
    dist_thres = cfg.get('dist_threshold', 100)
    max_gone = cfg.get('max_disappeared', 5)
    tracker = EuclideanDistTracker(dist_threshold=dist_thres, max_disappeared=max_gone)
    
    track_history = {}
    count_in, count_out = 0, 0
    line = cfg['line_coords'] 
    proc_w, proc_h = cfg['process_width'], cfg['process_height']

    # 4. Display & Warmup
    window_name = "Jetson Pro Counter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    if device.type != 'cpu':
        im_type = torch.half if use_fp16 else torch.float
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type(im_type))

    # Stats
    prev_time = time.time()
    fps = 0
    hw_cpu, hw_gpu = 0, 0
    frame_count = 0

    while True:
        # Đọc từ luồng riêng (Cực nhanh, không block)
        grabbed, frame_raw = cam.read()
        
        if not grabbed:
            if isinstance(src, str): break # Hết video
            else: continue # Lỗi cam, thử lại
        
        frame_count += 1
        
        # Resize nhẹ để xử lý vẽ
        frame = cv2.resize(frame_raw, (proc_w, proc_h))

        # --- AI INFERENCE ---
        img = letterbox(frame, imgsz, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if use_fp16 else img.float()
        img /= 255.0
        if len(img.shape) == 3: img = img[None]

        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, cfg['conf_thres'], cfg['iou_thres'], classes=cfg['classes'])
        
        # --- PROCESS ---
        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    detections.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2])-int(xyxy[0]), int(xyxy[3])-int(xyxy[1])])

        # --- TRACKING ---
        boxes_ids = tracker.update(detections)

        # --- DRAWING ---
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 2)

        for box_id in boxes_ids:
            x, y, w, h, id, cx, cy = box_id
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

            curr_point = (cx, cy)
            if id not in track_history:
                track_history[id] = [curr_point]
            else:
                track_history[id].append(curr_point)
                if len(track_history[id]) > 15: track_history[id].pop(0)

            if len(track_history[id]) >= 2:
                prev_point = track_history[id][-2]
                if intersect((line[0], line[1]), (line[2], line[3]), prev_point, curr_point):
                    cross_prod = vector_cross_product((line[0], line[1]), (line[2], line[3]), curr_point)
                    if cross_prod > 0: count_in += 1
                    else: count_out += 1
                    track_history[id] = []
                    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 4)

        # --- STATS ---
        t2 = time.time()
        fps = (fps * 0.95) + (1/(t2-prev_time) * 0.05)
        prev_time = t2
        
        if frame_count % 30 == 0: hw_cpu, hw_gpu = get_jetson_stats()

        cv2.rectangle(frame, (0, 0), (200, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"CPU: {hw_cpu}% GPU: {hw_gpu}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"IN: {count_in} | OUT: {count_out}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()