import sys
import time
import cv2
import torch
import numpy as np
from pathlib import Path
import os
import psutil  # <--- [MỚI] Thư viện đọc thông số hệ thống

# PyQt5 Imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QComboBox, QFileDialog, QGroupBox, QMessageBox, QGridLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

# Import YOLOv5 & Tracker
sys.path.append(str(Path(__file__).parent / 'yolov5'))
# Thêm try-except để bắt lỗi thiếu thư viện YOLOv5 v6.1
try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
    from utils.torch_utils import select_device
except ImportError:
    print("LỖI: Không tìm thấy folder 'yolov5'. Hãy đảm bảo bạn đã clone yolov5 v6.1 vào cùng thư mục.")
    sys.exit()

from JetsonYolo_Project.src.tracker import EuclideanDistTracker 

# --- [MỚI] HÀM ĐỌC THÔNG SỐ JETSON NANO ---
def get_jetson_stats():
    # 1. CPU & RAM Usage
    cpu_usage = psutil.cpu_percent()
    ram_info = psutil.virtual_memory()
    ram_usage = ram_info.percent
    
    # 2. GPU Load (Đọc từ file hệ thống Tegra)
    # File chứa tải GPU (0-1000), chia 10 để ra %
    gpu_usage = 0
    try:
        if os.path.exists("/sys/devices/gpu.0/load"):
            with open("/sys/devices/gpu.0/load", "r") as f:
                content = f.read().strip()
                gpu_usage = int(content) / 10.0
    except:
        gpu_usage = 0 
    return cpu_usage, gpu_usage, ram_usage

# --- HÀM TOÁN HỌC KIỂM TRA GIAO CẮT VECTOR (Giữ nguyên) ---
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def vector_cross_product(line_start, line_end, point):
    v_line = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
    v_point = np.array([point[0] - line_start[0], point[1] - line_start[1]])
    return np.cross(v_line, v_point)

# --- CLASS XỬ LÝ AI (Back-end) ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_info_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.weights = 'weights/yolov5s.pt'
        self.source = '0' 
        self.device_name = 'cuda:0' 
        self.img_size = 640
        self.conf_thres = 0.4
        self.iou_thres = 0.45
        
        # Line config (x1, y1, x2, y2)
        self.line = [1080, 1080, 1080, 0] 
        self.count_in = 0
        self.count_out = 0
        self.track_history = {} 

        # [MỚI] Biến tính Benchmark
        self.total_frames = 0
        self.total_infer_time = 0
        self.start_app_time = 0

    def run(self):
        # 1. Load Model (Cập nhật logic cho v6.1 & TensorRT)
        try:
            device = select_device(self.device_name.replace('cuda:0', '0'))
            is_engine = self.weights.endswith('.engine')

            # Load Model
            model = DetectMultiBackend(self.weights, device=device, dnn=False, data=None, fp16=False)
            stride = model.stride
            names = model.names
            
            # [MỚI] Tự động sửa Image Size nếu là Engine (Tránh lỗi mismatch)
            if is_engine:
                if hasattr(model, 'img_size'):
                    s = model.img_size
                    self.img_size = s[0] if isinstance(s, list) else s
                    print(f"-> Auto-update img_size to {self.img_size} form Engine metadata")
                half = False # Engine tự quản lý precision
            else:
                half = (device.type != 'cpu')
                if half: model.model.half()

            print(f"Loaded {self.weights} on {device} (Size: {self.img_size})")
        except Exception as e:
            self.error_signal.emit(f"Lỗi load model: {str(e)}")
            return

        # 2. Open Source
        src = self.source
        if src.isdigit(): src = int(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.error_signal.emit(f"Không thể mở nguồn video: {src}")
            return
        
        tracker = EuclideanDistTracker()

        # Warmup
        if device.type != 'cpu':
            model(torch.zeros(1, 3, self.img_size, self.img_size).to(device).type_as(next(model.model.parameters())))

        # [MỚI] Reset Benchmark stats
        self.total_frames = 0
        self.total_infer_time = 0
        self.start_app_time = time.time()
        hw_cpu, hw_gpu, hw_ram = 0, 0, 0

        while self._run_flag:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: break
            
            self.total_frames += 1

            # Pre-process
            # auto=not is_engine: Engine cần size cố định, PT cần auto stride
            img = letterbox(frame, self.img_size, stride=stride, auto=not is_engine)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            
            if is_engine:
                img = img.half() if model.fp16 else img.float()
            else:
                img = img.half() if half else img.float()
                
            img /= 255.0
            if len(img.shape) == 3: img = img[None]

            # Inference & Timer
            t1 = time.time()
            pred = model(img, augment=False, visualize=False)
            t2 = time.time()
            
            current_infer_ms = (t2 - t1) * 1000
            self.total_infer_time += current_infer_ms

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=[0]) 

            # Process Detections
            detections = []
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        detections.append([x1, y1, x2-x1, y2-y1])

            # Update Tracker
            boxes_ids = tracker.update(detections)

            # --- COUNTING LOGIC (Giữ nguyên logic của bạn) ---
            cv2.line(frame, (self.line[0], self.line[1]), (self.line[2], self.line[3]), (0, 255, 255), 3)
            # (Vẽ điểm A B để dễ căn chỉnh)
            cv2.putText(frame, "A", (self.line[0], self.line[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(frame, "B", (self.line[2], self.line[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            for box_id in boxes_ids:
                x, y, w, h, id, cx, cy = box_id
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

                curr_point = (cx, cy)
                if id not in self.track_history:
                    self.track_history[id] = [curr_point]
                else:
                    self.track_history[id].append(curr_point)
                    if len(self.track_history[id]) > 3: self.track_history[id].pop(0)

                if len(self.track_history[id]) >= 2:
                    prev_point = self.track_history[id][-2]
                    line_p1, line_p2 = (self.line[0], self.line[1]), (self.line[2], self.line[3])

                    if intersect(line_p1, line_p2, prev_point, curr_point):
                        cross_prod = vector_cross_product(line_p1, line_p2, curr_point)
                        if cross_prod > 0: self.count_in += 1
                        else: self.count_out += 1
                        self.track_history[id] = [] # Reset
                        cv2.line(frame, (self.line[0], self.line[1]), (self.line[2], self.line[3]), (0, 0, 255), 5)

            # --- [MỚI] TÍNH TOÁN BENCHMARK ---
            # Chỉ cập nhật thông số phần cứng mỗi 30 frame (để nhẹ máy)
            if self.total_frames % 30 == 0:
                hw_cpu, hw_gpu, hw_ram = get_jetson_stats()

            # Tính FPS trung bình (Average FPS)
            elapsed_time = time.time() - self.start_app_time
            avg_fps = self.total_frames / elapsed_time if elapsed_time > 0 else 0
            
            # Tính Infer time trung bình
            avg_infer = self.total_infer_time / self.total_frames if self.total_frames > 0 else 0
            
            # FPS tức thời
            fps_live = 1.0 / (time.time() - start_time)

            self.change_pixmap_signal.emit(frame)
            self.update_info_signal.emit({
                "fps_live": fps_live,
                "fps_avg": avg_fps,
                "infer_live": current_infer_ms,
                "infer_avg": avg_infer,
                "in": self.count_in, 
                "out": self.count_out,
                "cpu": hw_cpu,
                "gpu": hw_gpu,
                "ram": hw_ram
            })

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# --- CLASS GIAO DIỆN (Front-end) ---
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 Jetson Nano - Benchmark Tool")
        self.resize(1100, 750)
        self.thread = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # PANEL VIDEO
        video_layout = QVBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("background-color: #222; border: 2px solid #555;")
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.image_label)
        main_layout.addLayout(video_layout, stretch=3)

        # PANEL CONTROLS
        controls_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout, stretch=1)

        # 1. Config Model & Device
        grp_setup = QGroupBox("System Setup")
        l_setup = QVBoxLayout()
        
        self.btn_weights = QPushButton("Select Weights")
        self.btn_weights.clicked.connect(self.select_weights)
        self.lbl_weights = QLabel("yolov5s.pt")
        l_setup.addWidget(QLabel("Model Weights:"))
        l_setup.addWidget(self.lbl_weights)
        l_setup.addWidget(self.btn_weights)

        l_setup.addWidget(QLabel("Device:"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(["cuda:0", "cpu"]) 
        l_setup.addWidget(self.combo_device)

        l_setup.addWidget(QLabel("Video Source:"))
        src_layout = QHBoxLayout()
        self.input_source = QLineEdit("0")
        self.input_source.setPlaceholderText("Path or '0'")
        self.btn_browse = QPushButton("...")
        self.btn_browse.setMaximumWidth(40)
        self.btn_browse.clicked.connect(self.browse_video_file)
        
        src_layout.addWidget(self.input_source)
        src_layout.addWidget(self.btn_browse)
        l_setup.addLayout(src_layout)
        grp_setup.setLayout(l_setup)
        controls_layout.addWidget(grp_setup)

        # 2. Config Params
        grp_param = QGroupBox("Parameters")
        l_param = QVBoxLayout()
        l_param.addWidget(QLabel("Image Size:"))
        self.combo_size = QComboBox()
        self.combo_size.addItems(["320", "416", "640"])
        self.combo_size.setCurrentText("416")
        l_param.addWidget(self.combo_size)

        l_param.addWidget(QLabel("Line (x1,y1,x2,y2):"))
        self.input_line = QLineEdit("1080, 1080, 1080, 0") # Default line nằm ngang dễ test
        l_param.addWidget(self.input_line)
        self.btn_upd_line = QPushButton("Apply Line")
        self.btn_upd_line.clicked.connect(self.update_line_param)
        l_param.addWidget(self.btn_upd_line)
        grp_param.setLayout(l_param)
        controls_layout.addWidget(grp_param)

        # --- [MỚI] Bảng hiển thị Benchmark Hardware ---
        grp_hw = QGroupBox("Hardware Status")
        l_hw = QGridLayout()
        self.lbl_cpu = QLabel("CPU: 0%")
        self.lbl_gpu = QLabel("GPU: 0%")
        self.lbl_ram = QLabel("RAM: 0%")
        
        # Style đỏ/xanh cho nổi bật
        self.lbl_gpu.setStyleSheet("color: #d32f2f; font-weight: bold;") 
        
        l_hw.addWidget(self.lbl_cpu, 0, 0)
        l_hw.addWidget(self.lbl_gpu, 0, 1)
        l_hw.addWidget(self.lbl_ram, 1, 0)
        grp_hw.setLayout(l_hw)
        controls_layout.addWidget(grp_hw)

        # --- [MỚI] Bảng hiển thị Performance ---
        grp_stats = QGroupBox("Performance & Counting")
        l_stats = QVBoxLayout()
        
        self.lbl_fps_live = QLabel("Live FPS: --")
        self.lbl_fps_avg = QLabel("Avg FPS: --") # [MỚI]
        self.lbl_fps_avg.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")

        self.lbl_infer_live = QLabel("Infer Live: -- ms")
        self.lbl_infer_avg = QLabel("Infer Avg: -- ms") # [MỚI]

        l_stats.addWidget(self.lbl_fps_live)
        l_stats.addWidget(self.lbl_fps_avg)
        l_stats.addWidget(self.lbl_infer_live)
        l_stats.addWidget(self.lbl_infer_avg)
        
        l_stats.addWidget(QLabel("--- Counting ---"))
        self.lbl_count = QLabel("IN: 0  |  OUT: 0")
        self.lbl_count.setStyleSheet("color: blue; font-size: 18px; font-weight: bold; border: 1px solid #ccc; padding: 5px;")
        self.lbl_count.setAlignment(Qt.AlignCenter)
        l_stats.addWidget(self.lbl_count)
        
        grp_stats.setLayout(l_stats)
        controls_layout.addWidget(grp_stats)

        # Buttons
        self.btn_start = QPushButton("START BENCHMARK")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_video)
        controls_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self.stop_video)
        controls_layout.addWidget(self.btn_stop)

        controls_layout.addStretch()

    def select_weights(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Weights', '.', "Model (*.pt *.engine)")
        if fname:
            self.lbl_weights.setText(Path(fname).name)
            self.selected_weights = fname

    def browse_video_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Video', '.', "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if fname:
            self.input_source.setText(fname)

    def update_line_param(self):
        if self.thread:
            try:
                coords = [int(x.strip()) for x in self.input_line.text().split(',')]
                if len(coords) == 4:
                    self.thread.line = coords
            except: pass

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    # [MỚI] Cập nhật hàm này để hiển thị nhiều thông số hơn
    def update_stats(self, data):
        # Update Hardware
        self.lbl_cpu.setText(f"CPU: {data['cpu']}%")
        self.lbl_gpu.setText(f"GPU: {data['gpu']}%")
        self.lbl_ram.setText(f"RAM: {data['ram']}%")
        
        # Update Performance
        self.lbl_fps_live.setText(f"Live FPS: {data['fps_live']:.1f}")
        self.lbl_fps_avg.setText(f"Avg FPS: {data['fps_avg']:.2f}")
        self.lbl_infer_live.setText(f"Infer Live: {data['infer_live']:.1f} ms")
        self.lbl_infer_avg.setText(f"Infer Avg: {data['infer_avg']:.1f} ms")
        
        # Update Counting
        self.lbl_count.setText(f"IN: {data['in']}  |  OUT: {data['out']}")

    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.stop_video()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def start_video(self):
        self.thread = VideoThread()
        if hasattr(self, 'selected_weights'): 
            self.thread.weights = self.selected_weights
        
        self.thread.source = self.input_source.text()
        self.thread.device_name = self.combo_device.currentText()
        self.thread.img_size = int(self.combo_size.currentText())
        
        try:
            coords = [int(x.strip()) for x in self.input_line.text().split(',')]
            self.thread.line = coords
        except: pass

        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_info_signal.connect(self.update_stats)
        self.thread.error_signal.connect(self.show_error)
        self.thread.start()
        
        self.btn_start.setEnabled(False)
        self.input_source.setEnabled(False)
        self.combo_device.setEnabled(False)

    def stop_video(self):
        if self.thread:
            self.thread.stop()
        self.btn_start.setEnabled(True)
        self.input_source.setEnabled(True)
        self.combo_device.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())