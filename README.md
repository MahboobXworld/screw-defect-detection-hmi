# Screw Defect Detection HMI Dashboard

A premium, real-time Computer Vision & Human-Machine Interface (HMI) dashboard for automated industrial screw quality inspection. Powered by a custom-trained YOLOv8 segmentation model and advanced tracking, the system identifies, tracks, and classifies screws passing on a conveyor belt, providing real-time quality analytics, persistent databases, and exportable PDF/CSV reporting.

---

## 🚀 Key Features

* **Real-time Computer Vision Pipeline**: Integrated with a retrained YOLOv8s-seg model (trained for 100 epochs on the MVTec Screw dataset annotated via Label Studio) to detect screws and classify quality states:
  * `defect_head` (Index 0)
  * `defect_neck` (Index 1)
  * `screw` (Index 2, mapped internally to `good_screw` to preserve database and HMI compatibility)
  * `defect_thread` (Index 3)
  * `defect_tip` (Index 4)
* **Advanced Multi-Object Tracking & Counting**: A robust custom tracker combining centroid proximity and IoU matching to keep track of screw IDs across frames:
  * **Unified Screw-Level Detection Merging**: Automatically groups horizontally overlapping or vertically aligned detections belonging to the same physical screw (such as a good screw body detection and multiple defect markers), ensuring each screw is tracked as a single unified entity to prevent double-counting.
  * **Trajectory Smoothing via Kalman Filtering**: Integrates a 6-state constant-velocity Kalman Filter to predict states and smooth trajectories. If a screw is temporarily obscured or not detected for a frame or two due to glare, the tracker predicts its next position, keeping the track alive and preventing ID re-assignment.
  * **Conveyor Direction Auto-Detection**: Dynamically detects conveyor flow direction (Left-to-Right or Right-to-Left) by recording and averaging horizontal displacements.
  * **Entry-Zone Spawn Filtering & Counting Delay**: Restricts counting eligibility to screws spawning in the entry zone (outer 25% of the frame) and delays registration until they cross a center threshold (35% width), preventing middle-of-screen noise from inflating count statistics.
  * **Dynamic Label Correction**: Continually tracks classification history. If a screw is initially identified as "good" but later presents a defect near the center of the viewport, the system dynamically decrements the good count, increments the defective count, and updates the database record in-place.
* **Premium Dark Mode GUI**: A high-DPI scaling PyQt6 interface featuring:
  * Real-time camera feed visualization with color-coded bounding boxes.
  * Custom **Circular Progress Gauges** for real-time quality rates.
  * Multi-metric statistics charts showing total inspected, good, and defective counters.
  * Dynamic Progress Bars representing proportion breakdowns of different defects.
* **SQLite Persistent Logging**: Automatic insertion and updates of inspection results in a local SQLite database (`database/inspection.db`), allowing real-time statistical correction and traceability.
* **Automated Exporters**: Generate high-quality inspection reports with one click:
  * **CSV Reports**: Raw logs containing timestamp, defect type, confidence, and status.
  * **PDF Reports**: Publication-grade summaries containing visual statistics tables, quality scores, and defect breakdown percentages, rendered natively via PyQt6 `QPdfWriter`.

---

## 📁 Project Structure

```bash
HMI_0.1/
├── main.py                  # Main GUI entrypoint (handles High-DPI and stylesheet setup)
├── best.pt                  # Trained YOLOv8s-seg weights (100 epochs)
├── README.md                # Project documentation
├── core/                    # Core Python pipeline
│   ├── camera.py            # Video stream source receiver (file, camera, RTSP)
│   ├── detector.py          # YOLO model wrapper, class mapping, and prediction pipeline
│   ├── tracker.py           # Centroid/IoU/Kalman tracking and detection merging logic
│   ├── database.py          # SQLite database connection, schema setup, and updates
│   ├── statistics.py        # Real-time counter tracker and quality percentage calculator
│   └── exporter.py          # PDF & CSV reporting engine (built using QPdfWriter and Pandas)
├── ui/                      # PyQt6 User Interface components
│   ├── main_window.py       # Main GUI layout, video slots, and widget callbacks
│   ├── circular_gauge.py    # Custom glassmorphic circular progress widget
│   └── style.py             # Dark mode theme style definitions
├── database/                # SQLite local DB directory
│   └── inspection.db        # Persistent SQL inspection log
├── reports/                 # Output directory for exported PDF and CSV reports
├── snapshots/               # Snapshot images of detected defects
└── videos/                  # Conveyor test video directory
    └── output.mp4           # Default test video file
```

---

## 🛠️ Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with the required Python libraries:
```bash
pip install pyqt6 opencv-python ultralytics numpy pandas
```

### 2. Running the Dashboard
To start the dashboard interface:
```bash
python3 main.py
```

### 3. Running with a Custom Video/Camera
The video stream source can be configured directly through the dashboard UI. Select standard input files (such as `videos/output.mp4`) or connect live RTSP camera feeds.

---

## 📊 Analytics and Reporting
* **Inspection Database**: Logs are persistent and saved in `database/inspection.db`.
* **Exporting Reports**: Clicking "Export Report" in the GUI generates a PDF layout and a CSV datasheet inside the `reports/` folder.
* **Dynamic Corrections**: If a screw is initially identified as "good" but later presents a defect (e.g., thread defect) before exiting the screen, the system automatically decrements the good count, increments the defective count, updates the local database row using its original database ID, and refreshes the dashboard in real-time.
