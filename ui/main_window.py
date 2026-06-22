import sys
import os
import shutil
import time
import threading
from datetime import datetime
import numpy as np
import cv2
import torch


from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QSpinBox, QComboBox, QFrame, QScrollArea, QLineEdit,
    QMessageBox, QGraphicsOpacityEffect, QProgressBar, QApplication
)


from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, Qt, QPropertyAnimation
from PyQt6.QtGui import QImage, QPixmap, QColor

from ui.circular_gauge import CircularGauge
from ui.style import DARK_STYLE
from core.detector import Detector
from core.statistics import Statistics
from core.database import init_db, log_inspection, get_history, clear_history, update_inspection
from core.exporter import export_csv, export_pdf, export_excel
from core.tracker import IoUTracker, clean_detections

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class FreshFrameReader:
    """
    Constantly grabs frames from a VideoCapture source in a background thread
    to prevent frame buffering and keep the feed completely in real-time.
    """
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened() and sys.platform == "darwin":
            self.cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, name="FreshFrameReader", daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.01)
                continue
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=0.5)
        self.cap.release()


class InspectionWorker(QThread):
    """
    Worker thread that runs the webcam stream, video file, or loops through dataset images
    to run YOLOv8-Seg prediction in the background without blocking the HMI.
    """
    frame_processed = pyqtSignal(np.ndarray, object, str, float, np.ndarray)  # (annotated, results, label, conf, raw_frame)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    finished_automatically = pyqtSignal()

    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.running = False
        self.mode = "video"  # "camera" or "video"
        self.camera_source = 0
        self.video_path = ""
        self.restart_capture = False
        self.simulation_images = []
        self.sim_index = 0
        self.sim_interval = 1.5  # seconds
        self.paused = True
        self.step_triggered = False
        self.prev_iteration_time = 0.0

    def load_simulation_images(self, directory):
        """Scan directory for JPEG/PNG images."""
        if not os.path.isdir(directory):
            print(f"[Worker] Directory not found: {directory}")
            return
        self.simulation_images = sorted([
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        self.sim_index = 0
        print(f"[Worker] Loaded {len(self.simulation_images)} simulation images.")

    def trigger_step(self):
        """Request processing of exactly one frame (used when paused)."""
        self.step_triggered = True

    def stop(self):
        """Safely stop the thread loop."""
        self.running = False
        self.wait()

    def run(self):
        self.running = True
        cap = None
        last_time = time.time()
        frame_count = 0

        while self.running:
            # Reopen the capture interface if source changes
            if self.restart_capture:
                self.restart_capture = False
                if cap is not None:
                    cap.release()
                    cap = None

            t_start = None
            frame = None
            raw_frame = None
            is_new_frame = False

            # --- CAMERA MODE ---
            if self.mode == "camera":
                if cap is None or not cap.isOpened():
                    cap = FreshFrameReader(self.camera_source)
                    if not cap.isOpened():
                        self.error_occurred.emit("Failed to open camera source")
                        time.sleep(1)
                        continue
                
                # Check play/pause state in camera mode
                if self.paused and not self.step_triggered:
                    time.sleep(0.05)
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                raw_frame = frame.copy()
                is_new_frame = True
                
                if self.step_triggered:
                    self.step_triggered = False
                else:
                    # Run camera at ~30 FPS for smooth real-time viewing
                    time.sleep(0.03)

            # --- VIDEO MODE ---
            elif self.mode == "video":
                if not self.video_path:
                    time.sleep(0.1)
                    continue
                
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(self.video_path)
                    if not cap.isOpened():
                        self.error_occurred.emit("Failed to open video source")
                        time.sleep(1)
                        continue
                
                if self.paused and not self.step_triggered:
                    time.sleep(0.05)
                    continue
                
                t_start = time.time()
                
                fps_val = cap.get(cv2.CAP_PROP_FPS)
                if fps_val <= 0:
                    fps_val = 25.0
                frame_delay = 1.0 / fps_val
                
                # Skip frames to catch up with real-time if the previous prediction loop took too long
                if not self.step_triggered and self.prev_iteration_time > frame_delay:
                    skip_count = min(int(fps_val), int(self.prev_iteration_time * fps_val) - 1)
                    for _ in range(skip_count):
                        cap.grab()
                        
                ret, frame = cap.read()
                if not ret:
                    # Stop automatically when the video reaches the end
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.paused = True
                    self.finished_automatically.emit()
                    continue
                
                raw_frame = frame.copy()
                is_new_frame = True
                
                if self.step_triggered:
                    self.step_triggered = False
                elif self.prev_iteration_time <= frame_delay:
                    sleep_time = max(0.001, frame_delay - self.prev_iteration_time)
                    time.sleep(sleep_time)

            # --- SIMULATION MODE ---
            else:
                if cap is not None:
                    cap.release()
                    cap = None

                if not self.simulation_images:
                    time.sleep(0.5)
                    continue

                if self.paused and not self.step_triggered:
                    time.sleep(0.05)
                    continue

                image_path = self.simulation_images[self.sim_index]
                frame = cv2.imread(image_path)
                if frame is None:
                    self.sim_index = (self.sim_index + 1) % len(self.simulation_images)
                    time.sleep(0.1)
                    continue
                
                raw_frame = frame.copy()
                is_new_frame = True
                
                is_last_image = (self.sim_index == len(self.simulation_images) - 1)
                if self.step_triggered:
                    self.step_triggered = False
                    if is_last_image:
                        self.sim_index = 0
                        self.paused = True
                        self.finished_automatically.emit()
                    else:
                        self.sim_index += 1
                else:
                    time.sleep(self.sim_interval)
                    if is_last_image:
                        self.sim_index = 0
                        self.paused = True
                        self.finished_automatically.emit()
                    else:
                        self.sim_index += 1

            # --- PREDICTION AND SIGNAL EMISSION ---
            if is_new_frame and frame is not None:
                try:
                    # Run YOLOv8 prediction
                    results = self.detector.predict(frame)
                    label = self.detector.get_top_label(results)
                    
                    conf = 0.0
                    if label is not None:
                        boxes = results[0].boxes
                        if boxes is not None and len(boxes) > 0:
                            class_names = results[0].names
                            cls_ids = boxes.cls.cpu().numpy().astype(int)
                            confs = boxes.conf.cpu().numpy()
                            best_idx = -1
                            best_conf = -1.0
                            for idx, (cls_id, c) in enumerate(zip(cls_ids, confs)):
                                if class_names[cls_id] == label:
                                    if c > best_conf:
                                        best_conf = c
                                        best_idx = idx
                            if best_idx != -1:
                                conf = float(boxes.conf[best_idx].item())

                    # Draw segmentations/boxes
                    annotated = self.detector.draw_results(frame, results)
                    
                    # Emit frame details back to MainWindow
                    self.frame_processed.emit(annotated, results, label, conf, raw_frame)
                except Exception as e:
                    self.error_occurred.emit(f"YOLO Processing Error: {str(e)}")
                    time.sleep(0.1)

            # FPS calculation
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = frame_count / (now - last_time)
                self.fps_updated.emit(fps)
                frame_count = 0
                last_time = now

            if t_start is not None:
                self.prev_iteration_time = time.time() - t_start
            else:
                self.prev_iteration_time = 0.0

        if cap is not None:
            cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize sub-systems
        init_db()
        self.stats = Statistics()
        self.detector = Detector(model_path=os.path.join(PROJECT_ROOT, "models", "best.pt"))
        
        # System state tracking
        self.start_time = time.time()
        self.current_run_start_time = None
        
        # IoU Object Tracker
        self.tracker = IoUTracker(iou_threshold=0.3, max_lost_frames=30)
        
        # Setup Window Properties
        self.setWindowTitle("SCREW DEFECT INSPECTION SYSTEM HMI v1.0")
        
        # Determine screen dimensions dynamically to scale down on smaller laptops
        screen = QApplication.primaryScreen()
        use_fallback = True
        
        if screen:
            screen_geom = screen.availableGeometry()
            screen_w = screen_geom.width()
            screen_h = screen_geom.height()
            
            # Use dynamic sizing only if screen dimensions are valid and retrieveable
            if screen_w > 800 and screen_h > 600:
                use_fallback = False
                
                # Check if the screen is large enough to handle the comfortable defaults
                if screen_w >= 1680 and screen_h >= 1000:
                    # Large desktop screens: Use comfortable pre-adjusted default resolution
                    self.resize(1600, 920)
                    self.setMinimumSize(1280, 800)
                else:
                    # Smaller laptop screens: Scale down proportionally to fit the screen
                    init_w = min(1600, int(screen_w * 0.92))
                    init_h = min(920, int(screen_h * 0.90))
                    self.resize(init_w, init_h)
                    
                    # Set minimum boundary limits so controls don't overlap
                    min_w = min(1100, int(screen_w * 0.85))
                    min_h = min(720, int(screen_h * 0.85))
                    self.setMinimumSize(max(960, min_w), max(620, min_h))

        if use_fallback:
            # Safe default fallback sizing
            self.resize(1600, 920)
            self.setMinimumSize(1280, 800)
        
        # UI Assembly
        self.init_ui()
        self.apply_theme()
        
        # Background Worker Thread Configuration
        self.worker = InspectionWorker(self.detector)
        self.worker.frame_processed.connect(self.on_frame_processed)
        self.worker.fps_updated.connect(self.on_fps_updated)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.finished_automatically.connect(self.on_worker_finished_automatically)
        self.worker.start()
        
        # Timers
        self.sec_timer = QTimer(self)
        self.sec_timer.timeout.connect(self.on_second_timer)
        self.sec_timer.start(1000)
        
        # Load previous history into Table view
        self.load_history_from_db()
        self.populate_video_list()
        self.on_mode_changed(0)
        self.update_system_health()

    def apply_theme(self):
        self.setStyleSheet(DARK_STYLE)

    def init_ui(self):
        """Assembles all components of the Factory HMI."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # =========================================================================
        # 1. HEADER SECTION
        # =========================================================================
        header_frame = QFrame()
        header_frame.setObjectName("panel_frame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 10, 15, 10)
        
        # Left Title
        title_box = QVBoxLayout()
        title_lbl = QLabel("SCREW DEFECT INSPECTION SYSTEM v1.0")
        title_lbl.setObjectName("title_label")
        title_lbl.setStyleSheet("font-size: 18px; color: #FFFFFF; font-weight: bold;")
        sub_title_lbl = QLabel("AUTOMATED FACTORY HMI CORE")
        sub_title_lbl.setStyleSheet("font-size: 10px; color: #00BFFF; font-weight: bold; letter-spacing: 2px;")
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_title_lbl)
        header_layout.addLayout(title_box)
        
        header_layout.addStretch()
        
        # Center Status Badges
        status_box = QHBoxLayout()
        status_box.setSpacing(20)
        
        self.lbl_cam_status = QLabel("CAM: ONLINE")
        self.lbl_cam_status.setStyleSheet("font-size: 11px; font-weight: bold; color: #00FF88; border: 1px solid #00FF88; border-radius: 4px; padding: 4px 8px; background-color: #1A3E2B;")
        
        self.lbl_gpu_status = QLabel("GPU: ACTIVE")
        self.lbl_gpu_status.setStyleSheet("font-size: 11px; font-weight: bold; color: #00FF88; border: 1px solid #00FF88; border-radius: 4px; padding: 4px 8px; background-color: #1A3E2B;")
        
        self.lbl_storage_status = QLabel("STORAGE: OK")
        self.lbl_storage_status.setStyleSheet("font-size: 11px; font-weight: bold; color: #00FF88; border: 1px solid #00FF88; border-radius: 4px; padding: 4px 8px; background-color: #1A3E2B;")
        
        self.lbl_model_perf = QLabel("MODEL PERF: 0.0%")
        self.lbl_model_perf.setStyleSheet("font-size: 11px; font-weight: bold; color: #00FF88; border: 1px solid #00FF88; border-radius: 4px; padding: 4px 8px; background-color: #1A3E2B;")
        
        status_box.addWidget(self.lbl_cam_status)
        status_box.addWidget(self.lbl_gpu_status)
        status_box.addWidget(self.lbl_storage_status)
        status_box.addWidget(self.lbl_model_perf)
        header_layout.addLayout(status_box)
        
        header_layout.addSpacing(40)
        
        # Right Time & Date
        self.lbl_time = QLabel()
        self.lbl_time.setStyleSheet("font-family: monospace; font-size: 15px; color: #00BFFF; font-weight: bold;")
        self.lbl_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        header_layout.addWidget(self.lbl_time)
        
        main_layout.addWidget(header_frame, stretch=0)
        
        # =========================================================================
        # 2. DASHBOARD BODY (SPLIT LEFT / RIGHT)
        # =========================================================================
        body_layout = QHBoxLayout()
        body_layout.setSpacing(10)
        
        # -------------------------------------------------------------------------
        # 2.1 LEFT PANEL: CAMERA VIEW & CONTROLS
        # -------------------------------------------------------------------------
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        
        # Video/Image Container Frame
        cam_frame = QFrame()
        cam_frame.setObjectName("panel_frame")
        cam_vbox = QVBoxLayout(cam_frame)
        cam_vbox.setContentsMargins(8, 8, 8, 8)
        
        # Display Label
        self.image_label = QLabel("NO CAMERA SIGNAL")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #0A0A0A; border: 1px solid #2A2A2A; border-radius: 6px; font-size: 16px; font-weight: bold; color: #555555;")
        self.image_label.setMinimumSize(480, 360)
        cam_vbox.addWidget(self.image_label, stretch=1)
        
        left_layout.addWidget(cam_frame, stretch=1)
        
        # Control Panel Frame
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("panel_frame")
        ctrl_grid = QGridLayout(ctrl_frame)
        ctrl_grid.setContentsMargins(12, 12, 12, 12)
        ctrl_grid.setSpacing(10)
        
        # Mode Select Dropdown
        ctrl_grid.addWidget(QLabel("OPERATION MODE:"), 0, 0)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Video File Inspection",
            "Live Webcam Feed",
            "Mobile IP Camera Stream"
        ])
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        ctrl_grid.addWidget(self.combo_mode, 0, 1)
        
        # Controls Group
        self.btn_play_pause = QPushButton("RUN VIDEO")
        self.btn_play_pause.setObjectName("btn_success")
        self.btn_play_pause.clicked.connect(self.on_play_pause)
        ctrl_grid.addWidget(self.btn_play_pause, 0, 2)

        # Source Selection Controls (Row 1)
        self.lbl_camera_source_spin = QLabel("CAMERA INDEX:")
        self.spin_camera_source = QSpinBox()
        self.spin_camera_source.setRange(0, 9)
        self.spin_camera_source.setValue(0)
        self.spin_camera_source.valueChanged.connect(self.on_camera_source_changed)
        ctrl_grid.addWidget(self.lbl_camera_source_spin, 1, 0)
        ctrl_grid.addWidget(self.spin_camera_source, 1, 1)

        self.btn_upload_video = QPushButton("UPLOAD VIDEO 📤")
        self.btn_upload_video.clicked.connect(self.on_upload_video)
        self.combo_video_select = QComboBox()
        self.combo_video_select.currentIndexChanged.connect(self.on_video_selection_changed)
        ctrl_grid.addWidget(self.btn_upload_video, 1, 2)
        ctrl_grid.addWidget(self.combo_video_select, 1, 3)

        # Mobile IP Camera Stream controls (Row 2)
        self.lbl_mobile_url = QLabel("MOBILE STREAM URL:")
        self.txt_mobile_url = QLineEdit()
        self.txt_mobile_url.setPlaceholderText("e.g. http://192.168.0.X:8080/video or rtsp://...")
        self.txt_mobile_url.setText("http://192.168.0.197:8080/video")
        self.txt_mobile_url.textChanged.connect(self.on_mobile_url_changed)
        ctrl_grid.addWidget(self.lbl_mobile_url, 2, 0)
        ctrl_grid.addWidget(self.txt_mobile_url, 2, 1, 1, 3)

        # Export & Reset Group (Row 3)
        self.btn_reset = QPushButton("RESET STATS")
        self.btn_reset.setObjectName("btn_danger")
        self.btn_reset.clicked.connect(self.on_reset_stats)
        
        self.btn_export_csv = QPushButton("EXPORT CSV")
        self.btn_export_csv.clicked.connect(self.on_export_csv)
        
        self.btn_export_pdf = QPushButton("EXPORT PDF")
        self.btn_export_pdf.clicked.connect(self.on_export_pdf)
        
        self.btn_export_excel = QPushButton("EXPORT EXCEL")
        self.btn_export_excel.clicked.connect(self.on_export_excel)
        
        ctrl_grid.addWidget(self.btn_reset, 3, 0)
        ctrl_grid.addWidget(self.btn_export_csv, 3, 1)
        ctrl_grid.addWidget(self.btn_export_pdf, 3, 2)
        ctrl_grid.addWidget(self.btn_export_excel, 3, 3)
        
        left_layout.addWidget(ctrl_frame, stretch=0)
        body_layout.addLayout(left_layout, stretch=3)
        
        # -------------------------------------------------------------------------
        # 2.2 RIGHT PANEL: SCROLLABLE KPI CARDS
        # -------------------------------------------------------------------------
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(scroll_content)
        right_layout.setContentsMargins(0, 0, 5, 0)
        right_layout.setSpacing(10)
        
        # --- CARD 1: QUALITY GAUGE ---
        kpi_card = QFrame()
        kpi_card.setObjectName("card_frame")
        kpi_vbox = QVBoxLayout(kpi_card)
        kpi_vbox.setContentsMargins(10, 10, 10, 10)
        
        kpi_header = QLabel("Production Quality Gauge")
        kpi_header.setObjectName("card_header")
        kpi_vbox.addWidget(kpi_header)
        
        self.gauge_quality = CircularGauge(label="Quality", color=QColor("#FFC857"))
        kpi_vbox.addWidget(self.gauge_quality)
        right_layout.addWidget(kpi_card)
        
        # --- CARD 2: INSPECTION COUNTERS ---
        count_card = QFrame()
        count_card.setObjectName("card_frame")
        count_grid = QGridLayout(count_card)
        count_grid.setContentsMargins(12, 10, 12, 10)
        
        count_header = QLabel("Inspection Counter Summary")
        count_header.setObjectName("card_header")
        count_grid.addWidget(count_header, 0, 0, 1, 3)
        
        # Total
        lbl_tot_t = QLabel("TOTAL INSPECTED")
        lbl_tot_t.setObjectName("kpi_title")
        self.total_value = QLabel("0")
        self.total_value.setObjectName("kpi_value")
        count_grid.addWidget(lbl_tot_t, 1, 0)
        count_grid.addWidget(self.total_value, 2, 0)
        
        # Good
        lbl_gd_t = QLabel("GOOD SCREWS")
        lbl_gd_t.setObjectName("kpi_title")
        self.good_value = QLabel("0")
        self.good_value.setObjectName("kpi_value_good")
        count_grid.addWidget(lbl_gd_t, 1, 1)
        count_grid.addWidget(self.good_value, 2, 1)
        
        # Defective
        lbl_df_t = QLabel("DEFECTIVE PARTS")
        lbl_df_t.setObjectName("kpi_title")
        self.defective_value = QLabel("0")
        self.defective_value.setObjectName("kpi_value_defect")
        count_grid.addWidget(lbl_df_t, 1, 2)
        count_grid.addWidget(self.defective_value, 2, 2)
        
        right_layout.addWidget(count_card)

        # --- CARD 3: DEFECT BREAKDOWN & PROGRESS BARS ---
        breakdown_card = QFrame()
        breakdown_card.setObjectName("card_frame")
        breakdown_vbox = QVBoxLayout(breakdown_card)
        breakdown_vbox.setContentsMargins(12, 10, 12, 10)
        
        breakdown_header = QLabel("Defect Category Breakdown")
        breakdown_header.setObjectName("card_header")
        breakdown_vbox.addWidget(breakdown_header)
        
        # Grid of defect counters
        bd_grid = QGridLayout()
        bd_grid.setHorizontalSpacing(15)
        bd_grid.setVerticalSpacing(5)
        
        bd_grid.addWidget(QLabel("HEAD DEFECTS:"), 0, 0)
        self.head_value = QLabel("0")
        self.head_value.setStyleSheet("font-weight: bold; color: #FFC857;")
        bd_grid.addWidget(self.head_value, 0, 1)
        
        bd_grid.addWidget(QLabel("NECK DEFECTS:"), 0, 2)
        self.neck_value = QLabel("0")
        self.neck_value.setStyleSheet("font-weight: bold; color: #FFC857;")
        bd_grid.addWidget(self.neck_value, 0, 3)
        
        bd_grid.addWidget(QLabel("THREAD DEFECTS:"), 1, 0)
        self.thread_value = QLabel("0")
        self.thread_value.setStyleSheet("font-weight: bold; color: #FFC857;")
        bd_grid.addWidget(self.thread_value, 1, 1)
        
        bd_grid.addWidget(QLabel("TIP DEFECTS:"), 1, 2)
        self.tip_value = QLabel("0")
        self.tip_value.setStyleSheet("font-weight: bold; color: #FFC857;")
        bd_grid.addWidget(self.tip_value, 1, 3)
        
        breakdown_vbox.addLayout(bd_grid)
        
        # Progress bars to show percentage distribution of defects
        pb_layout = QVBoxLayout()
        pb_layout.setSpacing(4)
        
        self.lbl_head_pb = QLabel("Head Defects Proportion: 0%")
        self.lbl_head_pb.setStyleSheet("font-size: 10px; color: #A0A0A0;")
        self.pb_head = QProgressBar()
        self.pb_head.setStyleSheet("QProgressBar { height: 6px; border: none; background-color: #3A3A3A; border-radius: 3px; } QProgressBar::chunk { background-color: #FFC857; border-radius: 3px; }")
        self.pb_head.setTextVisible(False)
        
        self.lbl_neck_pb = QLabel("Neck Defects Proportion: 0%")
        self.lbl_neck_pb.setStyleSheet("font-size: 10px; color: #A0A0A0;")
        self.pb_neck = QProgressBar()
        self.pb_neck.setStyleSheet("QProgressBar { height: 6px; border: none; background-color: #3A3A3A; border-radius: 3px; } QProgressBar::chunk { background-color: #FFC857; border-radius: 3px; }")
        self.pb_neck.setTextVisible(False)

        self.lbl_thread_pb = QLabel("Thread Defects Proportion: 0%")
        self.lbl_thread_pb.setStyleSheet("font-size: 10px; color: #A0A0A0;")
        self.pb_thread = QProgressBar()
        self.pb_thread.setStyleSheet("QProgressBar { height: 6px; border: none; background-color: #3A3A3A; border-radius: 3px; } QProgressBar::chunk { background-color: #FFC857; border-radius: 3px; }")
        self.pb_thread.setTextVisible(False)

        self.lbl_tip_pb = QLabel("Tip Defects Proportion: 0%")
        self.lbl_tip_pb.setStyleSheet("font-size: 10px; color: #A0A0A0;")
        self.pb_tip = QProgressBar()
        self.pb_tip.setStyleSheet("QProgressBar { height: 6px; border: none; background-color: #3A3A3A; border-radius: 3px; } QProgressBar::chunk { background-color: #FFC857; border-radius: 3px; }")
        self.pb_tip.setTextVisible(False)
        
        pb_layout.addWidget(self.lbl_head_pb)
        pb_layout.addWidget(self.pb_head)
        pb_layout.addWidget(self.lbl_neck_pb)
        pb_layout.addWidget(self.pb_neck)
        pb_layout.addWidget(self.lbl_thread_pb)
        pb_layout.addWidget(self.pb_thread)
        pb_layout.addWidget(self.lbl_tip_pb)
        pb_layout.addWidget(self.pb_tip)
        
        breakdown_vbox.addLayout(pb_layout)
        
        right_layout.addWidget(breakdown_card)
        
        # (Trend chart removed) -- real-time defect trend chart has been disabled
        
        # --- CARD 5: LAST DEFECT SNAPSHOT ---
        defect_snap_card = QFrame()
        defect_snap_card.setObjectName("card_frame")
        ds_hbox = QHBoxLayout(defect_snap_card)
        ds_hbox.setContentsMargins(10, 10, 10, 10)
        
        snap_vbox = QVBoxLayout()
        snap_header = QLabel("Last Defect Snapshot")
        snap_header.setObjectName("card_header")
        snap_vbox.addWidget(snap_header)
        
        self.defect_snapshot_info = QLabel("No defects logged during session.")
        self.defect_snapshot_info.setStyleSheet("font-size: 11px; color: #A0A0A0; line-height: 1.3;")
        snap_vbox.addWidget(self.defect_snapshot_info)
        ds_hbox.addLayout(snap_vbox, stretch=2)
        
        self.defect_snapshot_label = QLabel("NO PREVIEW")
        self.defect_snapshot_label.setFixedSize(140, 100)
        self.defect_snapshot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.defect_snapshot_label.setStyleSheet("background-color: #121212; border: 1px solid #3A3A3A; border-radius: 4px; font-size: 9px; font-weight: bold; color: #444444;")
        ds_hbox.addWidget(self.defect_snapshot_label, stretch=1)
        
        right_layout.addWidget(defect_snap_card)
        
        # --- CARD 6: DEFECT HISTORY LOG (QTableWidget) ---
        history_card = QFrame()
        history_card.setObjectName("card_frame")
        hist_vbox = QVBoxLayout(history_card)
        hist_vbox.setContentsMargins(10, 10, 10, 10)
        
        hist_header = QLabel("Recent Defect Log")
        hist_header.setObjectName("card_header")
        hist_vbox.addWidget(hist_header)
        
        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(["TIME", "CLASSIFICATION", "CONF"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setMinimumHeight(150)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        hist_vbox.addWidget(self.history_table)
        
        right_layout.addWidget(history_card)

        # --- CARD 7: AI INSIGHTS PANEL ---
        ai_card = QFrame()
        ai_card.setObjectName("card_frame")
        ai_vbox = QVBoxLayout(ai_card)
        ai_vbox.setContentsMargins(12, 10, 12, 10)
        
        ai_header = QLabel("AI Diagnostics & Recommendations")
        ai_header.setObjectName("card_header")
        ai_vbox.addWidget(ai_header)
        
        ai_grid = QGridLayout()
        ai_grid.addWidget(QLabel("Primary Defect:"), 0, 0)
        self.ai_insight_defect = QLabel("NONE")
        self.ai_insight_defect.setStyleSheet("font-weight: bold; color: #00BFFF;")
        ai_grid.addWidget(self.ai_insight_defect, 0, 1)
        
        ai_grid.addWidget(QLabel("Frequency Proportion:"), 0, 2)
        self.ai_insight_freq = QLabel("0%")
        self.ai_insight_freq.setStyleSheet("font-weight: bold; color: #00BFFF;")
        ai_grid.addWidget(self.ai_insight_freq, 0, 3)
        ai_vbox.addLayout(ai_grid)
        
        self.ai_insight_rec = QLabel("All operations operating within parameters. No diagnostics generated.")
        self.ai_insight_rec.setWordWrap(True)
        self.ai_insight_rec.setStyleSheet("font-size: 11px; color: #FFC857; background-color: #2F2A1E; border: 1px solid #4D3F28; border-radius: 4px; padding: 6px; margin-top: 5px;")
        ai_vbox.addWidget(self.ai_insight_rec)
        
        right_layout.addWidget(ai_card)
        
        # --- CARD 8: RUNTIME & MODEL SPECIFICATIONS ---
        sys_card = QFrame()
        sys_card.setObjectName("card_frame")
        sys_grid = QGridLayout(sys_card)
        sys_grid.setContentsMargins(12, 10, 12, 10)
        
        sys_header = QLabel("System Performance & Metadata")
        sys_header.setObjectName("card_header")
        sys_grid.addWidget(sys_header, 0, 0, 1, 4)
        
        sys_grid.addWidget(QLabel("Runtime:"), 1, 0)
        self.lbl_runtime_val = QLabel("00:00:00")
        self.lbl_runtime_val.setStyleSheet("font-family: monospace; font-weight: bold;")
        sys_grid.addWidget(self.lbl_runtime_val, 1, 1)
        
        sys_grid.addWidget(QLabel("Speed (FPS):"), 1, 2)
        self.lbl_fps_val = QLabel("0.0")
        self.lbl_fps_val.setStyleSheet("font-family: monospace; font-weight: bold;")
        sys_grid.addWidget(self.lbl_fps_val, 1, 3)
        
        sys_grid.addWidget(QLabel("Device GPU:"), 2, 0)
        self.lbl_gpu_device = QLabel("Detecting...")
        self.lbl_gpu_device.setStyleSheet("font-weight: bold; color: #A0A0A0;")
        sys_grid.addWidget(self.lbl_gpu_device, 2, 1)

        sys_grid.addWidget(QLabel("Confidence Thresh:"), 2, 2)
        lbl_conf_limit = QLabel("0.40")
        lbl_conf_limit.setStyleSheet("font-weight: bold;")
        sys_grid.addWidget(lbl_conf_limit, 2, 3)
        
        right_layout.addWidget(sys_card)
        
        # Set content onto scrollbar container
        right_scroll.setWidget(scroll_content)
        body_layout.addWidget(right_scroll, stretch=2)
        
        main_layout.addLayout(body_layout, stretch=1)
        
        # =========================================================================
        # 3. ANIMATED STATUS BAR
        # =========================================================================
        self.status_bar_frame = QFrame()
        self.status_bar_frame.setStyleSheet("background-color: #1E1E1E; border-top: 1px solid #3A3A3A;")
        self.status_bar_frame.setFixedHeight(32)
        status_bar_layout = QHBoxLayout(self.status_bar_frame)
        status_bar_layout.setContentsMargins(10, 0, 10, 0)
        
        # Glow Overlay QFrame
        self.status_glow_overlay = QFrame(self.status_bar_frame)
        self.status_glow_overlay.setGeometry(0, 0, self.width(), 32)
        self.status_glow_overlay.setStyleSheet("background-color: transparent;")
        
        # Opacity effect for glow animation
        self.status_glow_effect = QGraphicsOpacityEffect(self.status_glow_overlay)
        self.status_glow_overlay.setGraphicsEffect(self.status_glow_effect)
        self.status_glow_effect.setOpacity(0.0)
        
        # Status Label text
        self.status_label = QLabel("✔ SYSTEM ONLINE | OPERATION TYPE: SIMULATION")
        self.status_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #A0A0A0;")
        status_bar_layout.addWidget(self.status_label)
        
        # QPropertyAnimation
        self.glow_anim = QPropertyAnimation(self.status_glow_effect, b"opacity")
        
        main_layout.addWidget(self.status_bar_frame, stretch=0)

    # =========================================================================
    # EVENT HANDLERS & SLOTS
    # =========================================================================
    @pyqtSlot(np.ndarray, object, str, float, np.ndarray)
    def on_frame_processed(self, annotated_frame, results, label, confidence, raw_frame):
        """Called when worker thread completes YOLO analysis on a new frame."""
        width = self.image_label.width()
        height = self.image_label.height()
        
        if width < 100: width = 640
        if height < 100: height = 480
        
        # Modes: index 0 -> Video File, index 1 -> Live Webcam
        # Photo/Dataset simulation modes removed; always use tracker-based counting.
        
        # Extract all YOLO detections satisfying thresholds (defects >= self.detector.confidence, good_screw >= 0.10)
        detections = []
        if results is not None and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                class_names = r.names
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                bboxes = r.boxes.xyxy.cpu().numpy()
                for idx, (cls_id, conf, bbox) in enumerate(zip(cls_ids, confs, bboxes)):
                    label_det = class_names[cls_id]
                    threshold = 0.10 if label_det == "good_screw" else self.detector.confidence
                    if conf >= threshold:
                        detections.append({
                            'bbox': bbox.tolist(),
                            'label': label_det,
                            'conf': float(conf)
                        })
                        
        # Update tracker and get active/retired tracks
        active_tracks, retired_tracks = self.tracker.update(detections)
        
        # Generate fresh annotated frame on GUI thread with tracking overlays
        annotated_frame = self.detector.draw_results(raw_frame, results, active_tracks)
        
        # Display prediction frame
        pixmap = self.bgr_to_pixmap(annotated_frame, width, height)
        self.image_label.setPixmap(pixmap)
        
        # Always update efficiency gauges on every incoming frame
        self.update_efficiency_ui()
        
        # Process new (uncounted) tracks immediately when they are active and crossing the counting boundary
        for track in active_tracks:
            if not track.counted:
                # Eligibility check: did it spawn near the edges (left, right, top, or bottom 25%)?
                # Counting boundary check: has it crossed the threshold from the entry side?
                # Fallback: automatically count any track present for at least 10 frames (stationary/center placement).
                h_img, w_img = raw_frame.shape[:2]
                spawn_x = track.cx_history[0]
                spawn_y = track.cy_history[0]
                curr_x = track.cx_history[-1]
                curr_y = track.cy_history[-1]
                
                presence_frames = len(track.cx_history)
                
                is_eligible = False
                should_count = False
                
                # At startup of stream (first 5 frames), count any detected screws immediately
                if self.tracker.frame_idx <= 5:
                    is_eligible = True
                    should_count = True
                elif presence_frames >= 10:
                    # Fallback for stationary or center-placed objects
                    is_eligible = True
                    should_count = True
                else:
                    # Check boundary entry points
                    if spawn_x < w_img * 0.25:  # Left entry
                        is_eligible = True
                        if curr_x >= w_img * 0.35:
                            should_count = True
                    elif spawn_x > w_img * 0.75:  # Right entry
                        is_eligible = True
                        if curr_x <= w_img * 0.65:
                            should_count = True
                    elif spawn_y < h_img * 0.25:  # Top entry
                        is_eligible = True
                        if curr_y >= h_img * 0.35:
                            should_count = True
                    elif spawn_y > h_img * 0.75:  # Bottom entry
                        is_eligible = True
                        if curr_y <= h_img * 0.65:
                            should_count = True
                        
                if not is_eligible:
                    # Mark as counted so we stop checking it
                    track.counted = True
                    continue
                    
                if not should_count:
                    # Eligible but hasn't crossed the threshold yet: check next frame
                    continue

                track.counted = True
                track.counted_label = track.label
                track.counted_conf = track.conf
                
                # Update stats
                self.stats.update(track.track_id, track.label)
                
                status = "GOOD" if track.label == "good_screw" else "DEFECT"
                image_path = None
                
                if status == "DEFECT":
                    # Save annotated frame showing segmentation masks/boxes/track ID
                    snapshot_dir = os.path.join(PROJECT_ROOT, "snapshots")
                    os.makedirs(snapshot_dir, exist_ok=True)
                    filename = f"defect_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats.defective}.jpg"
                    image_path = os.path.join(snapshot_dir, filename)
                    cv2.imwrite(image_path, annotated_frame)
                    
                    # Log Snapshot preview in UI
                    self.defect_snapshot_label.setPixmap(
                        self.bgr_to_pixmap(annotated_frame, self.defect_snapshot_label.width(), self.defect_snapshot_label.height())
                    )
                    self.defect_snapshot_info.setText(
                        f"TYPE: {track.label.replace('_', ' ').upper()}\n"
                        f"CONF: {track.conf*100:.1f}%\n"
                        f"TRACK ID: {track.track_id}\n"
                        f"TIME: {datetime.now().strftime('%H:%M:%S')}"
                    )
                    
                    # Add new record directly to top of History Table view
                    timestamp_str = datetime.now().strftime("%H:%M:%S")
                    self.history_table.insertRow(0)
                    self.history_table.setItem(0, 0, QTableWidgetItem(timestamp_str))
                    item_lbl = QTableWidgetItem(f"{track.label.replace('_', ' ').upper()} (ID: {track.track_id})")
                    item_lbl.setForeground(QColor("#FF3B5C"))
                    self.history_table.setItem(0, 1, item_lbl)
                    self.history_table.setItem(0, 2, QTableWidgetItem(f"{track.conf*100:.1f}%"))
                    
                # Log entry in SQL database
                track.db_row_id = log_inspection(track.label, track.conf, image_path, status)
                
                # Update stats text and progress bars
                self.update_stats_ui()
                
                # Trigger animated status bar glow
                self.trigger_status_glow(status, track.label)

        # Check retired tracks as well, in case they were never counted while active
        for track in retired_tracks:
            # We no longer count retired tracks that did not cross the boundary to prevent duplicate counts
            track.counted = True

        # Check for dynamic label changes in already counted active tracks
        for track in active_tracks:
            if track.counted and hasattr(track, 'counted_label') and track.counted_label != track.label:
                old_label = track.counted_label
                new_label = track.label
                track.counted_label = new_label
                track.counted_conf = track.conf
                
                # Update stats
                self.stats.update_label_change(track.track_id, old_label, new_label)
                
                status = "GOOD" if new_label == "good_screw" else "DEFECT"
                image_path = None
                
                if status == "DEFECT":
                    # Save annotated frame showing segmentation masks/boxes/track ID
                    snapshot_dir = os.path.join(PROJECT_ROOT, "snapshots")
                    os.makedirs(snapshot_dir, exist_ok=True)
                    filename = f"defect_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats.defective}.jpg"
                    image_path = os.path.join(snapshot_dir, filename)
                    cv2.imwrite(image_path, annotated_frame)
                    
                    # Log Snapshot preview in UI
                    self.defect_snapshot_label.setPixmap(
                        self.bgr_to_pixmap(annotated_frame, self.defect_snapshot_label.width(), self.defect_snapshot_label.height())
                    )
                    self.defect_snapshot_info.setText(
                        f"TYPE: {new_label.replace('_', ' ').upper()}\n"
                        f"CONF: {track.conf*100:.1f}%\n"
                        f"TRACK ID: {track.track_id}\n"
                        f"TIME: {datetime.now().strftime('%H:%M:%S')}"
                    )
                    
                    # Add new record directly to top of History Table view
                    timestamp_str = datetime.now().strftime("%H:%M:%S")
                    self.history_table.insertRow(0)
                    self.history_table.setItem(0, 0, QTableWidgetItem(timestamp_str))
                    item_lbl = QTableWidgetItem(f"{new_label.replace('_', ' ').upper()} (ID: {track.track_id})")
                    item_lbl.setForeground(QColor("#FF3B5C"))
                    self.history_table.setItem(0, 1, item_lbl)
                    self.history_table.setItem(0, 2, QTableWidgetItem(f"{track.conf*100:.1f}%"))
                
                # Update SQL DB record
                if hasattr(track, 'db_row_id') and track.db_row_id is not None:
                    update_inspection(track.db_row_id, new_label, track.conf, image_path, status)
                
                # Update stats text and progress bars
                self.update_stats_ui()
                
                # Trigger animated status bar glow
                self.trigger_status_glow(status, new_label)


        self.update_model_performance(confidence)

    @pyqtSlot(float)
    def on_fps_updated(self, fps):
        self.lbl_fps_val.setText(f"{fps:.1f}")

    @pyqtSlot(str)
    def on_worker_error(self, err_msg):
        print(f"[Worker Error] {err_msg}")
        self.status_label.setText(f"⚠ ERROR: {err_msg}")
        self.lbl_cam_status.setText("CAM: ERROR")
        self.lbl_cam_status.setStyleSheet("font-size: 11px; font-weight: bold; color: #FF3B5C; border: 1px solid #FF3B5C; border-radius: 4px; padding: 4px 8px; background-color: #3E1A1E;")

    @pyqtSlot()
    def on_worker_finished_automatically(self):
        """Triggered automatically when video file completes all frames."""
        if self.combo_mode.currentIndex() == 0:
            self.btn_play_pause.setText("RUN VIDEO")
            self.btn_play_pause.setObjectName("btn_success")
            self.status_label.setText("✔ VIDEO COMPLETED | ALL FRAMES PROCESSED")
            if not self.report_generated:
                self.auto_generate_reports()
        
        self.apply_theme()  # Repaint stylesheet to update button colors

    def update_model_performance(self, confidence):
        """Updates the performance badge to show the latest model confidence."""
        try:
            perf = float(confidence) * 100.0
        except Exception:
            perf = 0.0
        self.lbl_model_perf.setText(f"MODEL PERF: {perf:.1f}%")

    def on_second_timer(self):
        """Timer ticking every 1.0 second."""
        # 1. Update clock
        self.lbl_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 2. Update runtime duration
        runtime_sec = int(time.time() - self.start_time)
        hours = runtime_sec // 3600
        mins = (runtime_sec % 3600) // 60
        secs = runtime_sec % 60
        self.lbl_runtime_val.setText(f"{hours:02d}:{mins:02d}:{secs:02d}")
        
        # 3. Update indicators
        self.update_system_health()
        
        # 4. Update real-time efficiency gauges
        self.update_efficiency_ui()

    def update_system_health(self):
        """Evaluate hardware/storage health values and update indicator styles."""
        # Check GPU Availability details
        gpu_name, gpu_status, storage_status = get_system_specs()
        self.lbl_gpu_device.setText(gpu_name[:24]) # limit size
        
        # GPU Indicator
        if gpu_status == "ACTIVE":
            self.lbl_gpu_status.setText("GPU: ACTIVE")
            self.lbl_gpu_status.setStyleSheet("font-size: 11px; font-weight: bold; color: #00FF88; border: 1px solid #00FF88; border-radius: 4px; padding: 4px 8px; background-color: #1A3E2B;")
        else:
            self.lbl_gpu_status.setText("GPU: INACTIVE")
            self.lbl_gpu_status.setStyleSheet("font-size: 11px; font-weight: bold; color: #FFC857; border: 1px solid #FFC857; border-radius: 4px; padding: 4px 8px; background-color: #3E371A;")
            
        # Storage Indicator
        if storage_status == "OK":
            self.lbl_storage_status.setText("STORAGE: OK")
            self.lbl_storage_status.setStyleSheet("font-size: 11px; font-weight: bold; color: #00FF88; border: 1px solid #00FF88; border-radius: 4px; padding: 4px 8px; background-color: #1A3E2B;")
        else:
            self.lbl_storage_status.setText("STORAGE: LOW")
            self.lbl_storage_status.setStyleSheet("font-size: 11px; font-weight: bold; color: #FF3B5C; border: 1px solid #FF3B5C; border-radius: 4px; padding: 4px 8px; background-color: #3E1A1E;")
            
        # Camera Indicator
        if self.combo_mode.currentIndex() in (1, 2): # Webcam or Mobile Stream Mode
            if self.worker.running and not self.worker.paused:
                self.lbl_cam_status.setText("CAM: ONLINE" if self.combo_mode.currentIndex() == 1 else "CAM: STREAMING")
                self.lbl_cam_status.setStyleSheet("font-size: 11px; font-weight: bold; color: #00FF88; border: 1px solid #00FF88; border-radius: 4px; padding: 4px 8px; background-color: #1A3E2B;")
            else:
                self.lbl_cam_status.setText("CAM: OFFLINE")
                self.lbl_cam_status.setStyleSheet("font-size: 11px; font-weight: bold; color: #FFC857; border: 1px solid #FFC857; border-radius: 4px; padding: 4px 8px; background-color: #3E371A;")
        else:
            self.lbl_cam_status.setText("CAM: VIDEO")
            self.lbl_cam_status.setStyleSheet("font-size: 11px; font-weight: bold; color: #00BFFF; border: 1px solid #00BFFF; border-radius: 4px; padding: 4px 8px; background-color: #1A343E;")

    def update_stats_ui(self):
        """Refreshes text values, progress bars, circular gauges, and plot."""
        # Counters
        self.total_value.setText(str(self.stats.total))
        self.good_value.setText(str(self.stats.good))
        self.defective_value.setText(str(self.stats.defective))
        
        self.head_value.setText(str(self.stats.head))
        self.neck_value.setText(str(self.stats.neck))
        self.thread_value.setText(str(self.stats.thread))
        self.tip_value.setText(str(self.stats.tip))
        
        # Update real-time efficiency gauges
        self.update_efficiency_ui()
        
        # Update progress bars (percentage of defects)
        defects = self.stats.defective
        if defects > 0:
            self.pb_head.setValue(int(self.stats.head / defects * 100))
            self.lbl_head_pb.setText(f"Head Defects Proportion: {int(self.stats.head / defects * 100)}%")
            
            self.pb_neck.setValue(int(self.stats.neck / defects * 100))
            self.lbl_neck_pb.setText(f"Neck Defects Proportion: {int(self.stats.neck / defects * 100)}%")
            
            self.pb_thread.setValue(int(self.stats.thread / defects * 100))
            self.lbl_thread_pb.setText(f"Thread Defects Proportion: {int(self.stats.thread / defects * 100)}%")
            
            self.pb_tip.setValue(int(self.stats.tip / defects * 100))
            self.lbl_tip_pb.setText(f"Tip Defects Proportion: {int(self.stats.tip / defects * 100)}%")
        else:
            for pb in [self.pb_head, self.pb_neck, self.pb_thread, self.pb_tip]:
                pb.setValue(0)
            self.lbl_head_pb.setText("Head Defects Proportion: 0%")
            self.lbl_neck_pb.setText("Neck Defects Proportion: 0%")
            self.lbl_thread_pb.setText("Thread Defects Proportion: 0%")
            self.lbl_tip_pb.setText("Tip Defects Proportion: 0%")



        # 5. Update AI Insights diagnostics
        defect_type, freq, rec = get_ai_insight(self.stats)
        self.ai_insight_defect.setText(defect_type)
        self.ai_insight_freq.setText(freq)
        self.ai_insight_rec.setText(rec)

    def auto_generate_reports(self):
        """Automatically exports CSV, PDF, and Excel report files once when the run finishes."""
        try:
            history = get_history(100, since=self.current_run_start_time) if self.current_run_start_time else get_history(100)
            csv_file = export_csv(self.stats, history)
            pdf_file = export_pdf(self.stats, history)
            excel_file = export_excel(self.stats, history)
            print(f"[Auto Report] Generated {csv_file}, {pdf_file}, and {excel_file}")
        except Exception as e:
            print(f"[Auto Report Error] {e}")
        self.report_generated = True

    def update_efficiency_ui(self):
        """Calculates Quality and updates the single quality gauge in real-time."""
        quality_rate = self.stats.rolling_yield_percent()
        self.gauge_quality.setValue(quality_rate)


    def trigger_status_glow(self, status, label):
        """Pulses status bar overlay red or green using QPropertyAnimation."""
        self.glow_anim.stop()
        
        # Resize glow overlay geometry to align with active window
        self.status_glow_overlay.setGeometry(0, 0, self.status_bar_frame.width(), 32)
        
        if status == "GOOD":
            self.status_glow_overlay.setStyleSheet("background-color: #00FF88;")
            self.status_label.setText(f"✔ INSPECTION OK: GOOD SCREW DETECTED")
            
            # Simple fade animation (fade-in then fade-out)
            self.glow_anim.setDuration(1200)
            self.glow_anim.setStartValue(0.7)
            self.glow_anim.setKeyValueAt(0.5, 0.7)
            self.glow_anim.setEndValue(0.0)
            self.glow_anim.setLoopCount(1)
            self.glow_anim.start()
        else:
            self.status_glow_overlay.setStyleSheet("background-color: #FF3B5C;")
            defect_readable = label.replace('_', ' ').upper()
            self.status_label.setText(f"⚠ INSPECTION DEFECT: {defect_readable} DETECTED")
            
            # Repetitive warning pulse
            self.glow_anim.setDuration(1500)
            self.glow_anim.setStartValue(0.8)
            self.glow_anim.setKeyValueAt(0.5, 0.1)
            self.glow_anim.setEndValue(0.8)
            self.glow_anim.setLoopCount(-1) # infinite looping until next trigger
            self.glow_anim.start()

    def load_history_from_db(self):
        """Populates the Log Table view from Sqlite database on startup."""
        try:
            records = get_history(50)
            self.history_table.setRowCount(0)
            
            for i, r in enumerate(records):
                self.history_table.insertRow(i)
                
                # Fetch time component from timestamp
                dt = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
                time_str = dt.strftime("%H:%M:%S")
                
                self.history_table.setItem(i, 0, QTableWidgetItem(time_str))
                
                item_lbl = QTableWidgetItem(r["defect_type"].replace('_', ' ').upper())
                if r["status"] == "DEFECT":
                    item_lbl.setForeground(QColor("#FF3B5C"))
                else:
                    item_lbl.setForeground(QColor("#00FF88"))
                
                self.history_table.setItem(i, 1, item_lbl)
                self.history_table.setItem(i, 2, QTableWidgetItem(f"{r['confidence']*100:.1f}%"))
                
                # Populate last defect image if available in historical logs
                if r["status"] == "DEFECT" and r["image_path"] and os.path.exists(r["image_path"]):
                    self.defect_snapshot_label.setPixmap(
                        self.bgr_to_pixmap(cv2.imread(r["image_path"]), self.defect_snapshot_label.width(), self.defect_snapshot_label.height())
                    )
                    self.defect_snapshot_info.setText(
                        f"TYPE: {r['defect_type'].replace('_', ' ').upper()}\nCONF: {r['confidence']*100:.1f}%\nTIME: {time_str}"
                    )
        except Exception as e:
            print(f"[DB History Load Error] {e}")

    # =========================================================================
    # ACTIONS: COMBOS, BUTTONS & SLIDER SLOTS
    # =========================================================================
    def on_mode_changed(self, idx):
        """Adjusts worker thread mode selection and updates source UI controls."""
        # Force the worker to close any active capture device
        self.worker.restart_capture = True
        self.worker.paused = True
        self.report_generated = False
        self.current_run_start_time = None
        
        # Reset debouncer states when changing modes
        self.screw_in_view = False
        self.consecutive_empty_frames = 0
        self.last_cx = None
        self.last_detection_time = 0.0
        self.tracker.reset()
        
        # Hide all source settings by default
        self.lbl_camera_source_spin.hide()
        self.spin_camera_source.hide()
        self.btn_upload_video.hide()
        self.combo_video_select.hide()
        self.lbl_mobile_url.hide()
        self.txt_mobile_url.hide()
        
        self.report_generated = False
        if idx == 0:  # Video File Inspection
            self.worker.mode = "video"
            self.worker.paused = True
            self.btn_play_pause.setEnabled(True)
            self.btn_play_pause.setText("RUN VIDEO")
            self.btn_play_pause.setObjectName("btn_success")
            self.status_label.setText("✔ VIDEO MODE READY | PRESS START")
            
            # Show Video controls
            self.btn_upload_video.show()
            self.combo_video_select.show()
        elif idx == 1:  # Live Webcam Feed
            self.worker.mode = "camera"
            self.worker.camera_source = self.spin_camera_source.value()
            self.worker.paused = True
            self.btn_play_pause.setEnabled(True)
            self.btn_play_pause.setText("START CAMERA")
            self.btn_play_pause.setObjectName("btn_success")
            self.status_label.setText("✔ CAMERA MODE READY | PRESS START")
            
            # Show Camera index selector
            self.lbl_camera_source_spin.show()
            self.spin_camera_source.show()
        elif idx == 2:  # Mobile IP Camera Stream
            self.worker.mode = "camera"
            self.worker.camera_source = self.txt_mobile_url.text().strip()
            self.worker.paused = True
            self.btn_play_pause.setEnabled(True)
            self.btn_play_pause.setText("START STREAM")
            self.btn_play_pause.setObjectName("btn_success")
            self.status_label.setText("✔ MOBILE STREAM READY | PRESS START")
            
            # Show Mobile Stream URL controls
            self.lbl_mobile_url.show()
            self.txt_mobile_url.show()
            
        self.apply_theme()
        self.update_system_health()

    def on_play_pause(self):
        """Toggles execution of worker thread frame grabbing loop."""
        self.worker.paused = not self.worker.paused
        if self.worker.paused:
            if self.combo_mode.currentIndex() == 0:
                self.btn_play_pause.setText("RUN VIDEO")
                self.status_label.setText("✔ VIDEO STOPPED")
                if self.current_run_start_time and not self.report_generated:
                    self.auto_generate_reports()
            elif self.combo_mode.currentIndex() == 1:
                self.btn_play_pause.setText("START CAMERA")
                self.worker.restart_capture = True
                self.clear_inspection_view("NO CAMERA SIGNAL")
                self.status_label.setText("✔ CAMERA STOPPED")
                if not self.report_generated:
                    self.auto_generate_reports()
            elif self.combo_mode.currentIndex() == 2:
                self.btn_play_pause.setText("START STREAM")
                self.worker.restart_capture = True
                self.clear_inspection_view("NO MOBILE STREAM SIGNAL")
                self.status_label.setText("✔ STREAM STOPPED")
                if not self.report_generated:
                    self.auto_generate_reports()
            self.btn_play_pause.setObjectName("btn_success")
        else:
            if self.combo_mode.currentIndex() == 1:
                self.btn_play_pause.setText("STOP CAMERA")
            elif self.combo_mode.currentIndex() == 2:
                self.btn_play_pause.setText("STOP STREAM")
                self.worker.camera_source = self.txt_mobile_url.text().strip()
            else:
                self.btn_play_pause.setText("PAUSE RUN")
            self.btn_play_pause.setObjectName("btn_danger")
            self.report_generated = False
            self.current_run_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.apply_theme() # trigger repaint stylesheet update for btn colors
        self.update_system_health()

    def clear_inspection_view(self, message="NO CAMERA SIGNAL"):
        """Clears the inspection display and renders a placeholder message."""
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(message)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def on_camera_source_changed(self, val):
        """Adjusts selected camera hardware port."""
        self.worker.camera_source = val
        self.worker.restart_capture = True
        self.status_label.setText(f"✔ CAMERA INDEX UPDATED TO {val}")

    def on_mobile_url_changed(self, val):
        """Adjusts selected mobile camera stream URL source."""
        self.worker.camera_source = val.strip()
        self.worker.restart_capture = True
        self.status_label.setText(f"✔ MOBILE STREAM URL UPDATED")

    def populate_video_list(self):
        """Scans the videos directory and populates the video dropdown list."""
        video_dir = os.path.join(PROJECT_ROOT, "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        self.combo_video_select.blockSignals(True)
        self.combo_video_select.clear()
        
        # Scan for video files
        videos = sorted([
            f for f in os.listdir(video_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))
        ])
        
        if videos:
            self.combo_video_select.addItems(videos)
            # Set default path to first video
            first_video_path = os.path.join(video_dir, videos[0])
            self.worker.video_path = first_video_path
        else:
            self.combo_video_select.addItem("No videos uploaded")
            self.worker.video_path = ""
            
        self.combo_video_select.blockSignals(False)

    def on_upload_video(self):
        """Prompts user to select a video file and uploads (copies) it to videos/ directory."""
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video to Upload", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        if file_path:
            filename = os.path.basename(file_path)
            dest_dir = os.path.join(PROJECT_ROOT, "videos")
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, filename)
            
            try:
                # Copy file to the project's videos/ folder
                shutil.copy2(file_path, dest_path)
                
                # Refresh dropdown
                self.populate_video_list()
                
                # Select the newly uploaded video
                idx = self.combo_video_select.findText(filename)
                if idx >= 0:
                    self.combo_video_select.setCurrentIndex(idx)
                
                QMessageBox.information(
                    self, "Video Uploaded", 
                    f"Successfully uploaded and loaded video:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Upload Error", f"Failed to upload video: {str(e)}")

    def on_video_selection_changed(self, idx):
        """Called when user selects a different video from dropdown."""
        current_text = self.combo_video_select.itemText(idx)
        if current_text and current_text != "No videos uploaded":
            video_dir = os.path.join(PROJECT_ROOT, "videos")
            file_path = os.path.join(video_dir, current_text)
            self.worker.video_path = file_path
            self.worker.restart_capture = True
            self.status_label.setText(f"✔ VIDEO SOURCE LOADED: {current_text}")
        else:
            self.worker.video_path = ""
            self.status_label.setText("⚠ NO VIDEO SOURCE LOADED")

    def on_reset_stats(self):
        """Confirm, clear statistical widgets, clear plot series, and empty database tables."""
        reply = QMessageBox.question(
            self, "Reset Statistics", 
            "Are you sure you want to clear session statistics and database logs?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.stats.reset()
            clear_history()
            
            # Reset debouncer states
            self.screw_in_view = False
            self.consecutive_empty_frames = 0
            self.last_cx = None
            self.last_detection_time = 0.0
            
            # Trend chart removed; nothing to clear
            
            # Reset widgets
            self.history_table.setRowCount(0)
            self.defect_snapshot_label.setPixmap(QPixmap())
            self.defect_snapshot_label.setText("NO PREVIEW")
            self.defect_snapshot_info.setText("No defects logged during session.")
            
            self.update_stats_ui()
            self.status_label.setText("✔ STATISTICS AND HISTORY RESET SUCCESSFUL")

    def on_export_csv(self):
        try:
            history = get_history(100, since=self.current_run_start_time) if self.current_run_start_time else get_history(100)
            filename = export_csv(self.stats, history)
            QMessageBox.information(self, "Export Report", f"Successfully saved session summary to CSV:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {str(e)}")

    def on_export_pdf(self):
        try:
            history = get_history(100, since=self.current_run_start_time) if self.current_run_start_time else get_history(100)
            filename = export_pdf(self.stats, history)
            QMessageBox.information(self, "Export Report", f"Successfully printed session report to PDF:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export PDF: {str(e)}")

    def on_export_excel(self):
        try:
            history = get_history(100, since=self.current_run_start_time) if self.current_run_start_time else get_history(100)
            filename = export_excel(self.stats, history)
            QMessageBox.information(self, "Export Report", f"Successfully saved session summary to Excel:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export Excel: {str(e)}")

    # =========================================================================
    # HELPERS
    # =========================================================================
    def bgr_to_pixmap(self, bgr_image, target_width, target_height):
        """Converts BGR NumPy Array (OpenCV format) to QPixmap scaled to match target dimensions."""
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        return pixmap.scaled(
            target_width, target_height, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )

    def closeEvent(self, event):
        """Ensure background worker thread completes safely before exit."""
        self.worker.stop()
        self.sec_timer.stop()
        event.accept()


# Auxiliary system functions
def get_system_specs():
    """Reads system stats: GPU acceleration device and Storage thresholds."""
    gpu_name = "CPU"
    gpu_status = "INACTIVE"
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_status = "ACTIVE"
    elif torch.backends.mps.is_available():
        gpu_name = "Apple Silicon GPU (MPS)"
        gpu_status = "ACTIVE"
        
    total, used, free = shutil.disk_usage(".")
    storage_status = "OK" if (free / total) > 0.1 else "LOW"
    
    return gpu_name, gpu_status, storage_status


def get_ai_insight(stats):
    """Diagnoses stats and provides corrective factory recommendations."""
    most_common = stats.most_common_defect()
    if not most_common:
        return "NONE", "0%", "All inspections within nominal parameters. No diagnostics generated."
        
    total_defects = stats.defective
    # maps head_defect -> head, tip_defect -> tip
    defect_key = most_common.split("_")[0]
    count = getattr(stats, defect_key, 0)
    freq = f"{round((count / total_defects * 100) if total_defects > 0 else 0)}%"
    
    defect_readable = most_common.replace("_", " ").upper()
    
    if most_common == "head_defect":
        rec = "Head defect frequency is critical. Calibrate wire feeding tension, adjust punch stroke alignment, and inspect header forming dies for micro-cracks."
    elif most_common == "neck_defect":
        rec = "Elevated neck defects. Review workholding chuck alignment, adjust clamp mechanical pressure, and inspect blank stock material shear hardness."
    elif most_common == "thread_defect":
        rec = "Thread rolling defects dominate. Inspect roll dies for flat spot degradation, optimize thread cutting coolant flow, and measure blank core diameter."
    elif most_common == "tip_defect":
        rec = "Tip defects detected. Inspect cutoff knives, verify shear cutter alignment, and check tip chamfering tool tip clearance."
    else:
        rec = "Inspect conveyor vibration and sensor calibration."
        
    return defect_readable, freq, rec