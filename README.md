# Screw Defect Detection HMI Dashboard

A premium, real-time Computer Vision & Human-Machine Interface (HMI) dashboard for automated industrial screw quality inspection. Powered by YOLO segmentation model and object tracking, the system identifies, tracks, and classifies screws passing on a conveyor belt, providing real-time quality analytics, persistent databases, and exportable PDF/CSV reporting.

---

## 🚀 Key Features

* **Real-time Computer Vision Pipeline**: Integrated with a retrained YOLOv8 segmentation model to detect screws and classify quality states:
  * `head_defect` (Index 0)
  * `neck_defect` (Index 1)
  * `screw` (Index 2, mapped internally to `good_screw` to preserve database and HMI compatibility)
  * `thread_defect` (Index 3)
  * `tip_defect` (Index 4)
* **Advanced Multi-Object Tracking**: Centroid and IoU-based multi-screw tracker containing:
  * **Screw-Level Detection Merging**: Automatically groups horizontally overlapping or vertically aligned detections belonging to the same physical screw (such as a good screw body detection and multiple defect markers), ensuring each screw is tracked as a single unified entity.
  * **Conveyor Direction Auto-Detection**: Dynamically detects conveyor flow direction (Left-to-Right or Right-to-Left) based on screw movement history.
  * **Spurious Noise / Spawn Filtering**: Restricts counting eligibility to screws spawning near the conveyor entrance, preventing re-detected lost tracks or middle-of-screen noise from double-counting.
* **Premium Dark Mode GUI**: A high-DPI scaling PyQt6 interface featuring:
  * Real-time camera feed visualization with color-coded bounding boxes.
  * Custom **Circular Progress Gauges** for real-time quality rates.
  * Multi-metric statistics charts showing total inspected, good, and defective counters.
  * Dynamic Progress Bars representing proportion breakdowns of different defects.
* **SQLite Persistent Logging**: Automatic insertion and updates of inspection results in a local SQLite database (`inspection.db`), allowing real-time statistical correction if a screw's classification changes dynamically as it crosses the frame.
* **Automated Exporters**: Generate high-quality inspection reports with one click:
  * **CSV Reports**: Raw logs containing timestamp, defect type, confidence, and status.
  * **PDF Reports**: Publication-grade summaries containing visual statistics tables, quality scores, and defect breakdown percentages.

---

## 📁 Project Structure

```bash
HMI_0.1/
├── main.py                  # Main GUI entrypoint (handles High-DPI and stylesheet setup)
├── webcam.py                # Optional webcam-specific runtime script
├── best.pt                  # Trained YOLOv8 weights for screw defect detection
├── README.md                # Project documentation
├── core/                    # Core Python pipeline
│   ├── camera.py            # Video stream source receiver (file, camera, RTSP)
│   ├── detector.py          # YOLO model wrapper and prediction pipeline
│   ├── tracker.py           # Centroid/IoU tracking and detection clean-up logic
│   ├── database.py          # SQLite database connection, schema setup, and updates
│   ├── statistics.py        # Real-time counter tracker and quality percentage calculator
│   └── exporter.py          # PDF & CSV reporting engine
├── ui/                      # PyQt6 User Interface components
│   ├── main_window.py       # Main GUI layout, video slots, and widget callbacks
│   ├── circular_gauge.py    # Custom glassmorphic circular progress widget
│   └── style.py             # Dark mode theme style definitions
├── database/                # SQLite local DB directory
│   └── inspection.db        # Persistent SQL inspection log
├── reports/                 # Output directory for exported PDF and CSV reports
└── snapshots/               # Snapshot images of detected defects
```

---

## 🛠️ Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with the required system packages:
```bash
pip install pyqt6 opencv-python ultralytics numpy reportlab pandas
```

### 2. Running the Dashboard
To start the dashboard interface:
```bash
python3 main.py
```

### 3. Running with a Custom Video/Camera
The video stream source can be configured directly through the dashboard UI. Select standard input files (e.g., `.mp4` videos located in `videos/`) or connect live RTSP cameras.

---

## 📊 Analytics and Reporting
* **Inspection Database**: Logs are persistent and saved in `database/inspection.db`.
* **Exporting Reports**: Clicking "Export Report" in the GUI generates a PDF layout and a CSV datasheet inside the `reports/` folder.
* **Dynamic Corrections**: If a screw is initially identified as "good" but later presents a defect (e.g., thread defect) before exiting the screen, the system automatically decrements the good count, increments the defective count, updates the local database row, and refreshes the dashboard in real-time.
