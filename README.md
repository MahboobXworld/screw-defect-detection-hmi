# Screw Defect Detection HMI Dashboard

A premium, real-time Computer Vision & Human-Machine Interface (HMI) dashboard for automated industrial screw quality inspection. Powered by a custom-trained YOLOv8 segmentation model and advanced tracking, the system identifies, tracks, and classifies screws passing on a conveyor belt, providing real-time quality analytics, persistent databases, and exportable PDF/CSV reporting.

---

## 🚀 Key Features

* **Real-time Computer Vision Pipeline**: Integrated with a retrained YOLOv8s-seg model (trained for 100 epochs on the MVTec Screw dataset annotated via Label Studio) to detect screws and classify quality states:
  * `defect_head` (Index 0)
  * `defect_neck` (Index 1)
  * `screw` (Index 2)
  * `defect_thread` (Index 3)
  * `defect_tip` (Index 4)
* **Advanced Multi-Object Tracking & Counting**: A robust custom tracker combining centroid proximity and IoU matching to keep track of screw IDs across frames:
  * **Unified 2D Isotropic Detection Merging**: Automatically groups overlapping or vertically/diagonally aligned detections belonging to the same physical screw (such as a good screw body detection and multiple defect markers) using 2D isotropic proximity rules, ensuring each screw is tracked as a single unified entity to prevent double-counting.
  * **Trajectory Smoothing via Kalman Filtering**: Integrates a 6-state constant-velocity Kalman Filter to predict states and smooth trajectories. If a screw is temporarily obscured or not detected for a frame or two due to glare, the tracker predicts its next position, keeping the track alive and preventing ID re-assignment.
  * **Direction-Agnostic & Drift-Resistant Matching**: Track-to-detection matching compares new detections against **both** the Kalman-predicted states and the last-seen positions. This allows tracking objects moving in arbitrary directions (horizontal, vertical, diagonal) and eliminates ID switching for stationary objects whose Kalman filter states drift.
  * **Multi-Directional Boundary Entry & Stationary Count Fallback**: Handles entry zones along all four sides of the frame (Left, Right, Top, Bottom) to dynamically register screw crossings. Incorporates a stationary presence fallback where any screw present in the frame for at least 10 frames is automatically counted, making the system compatible with stationary/stop-and-go conveyor belts.
  * **Dynamic Label Correction**: Continually tracks classification history. If a screw is initially identified as "good" but later presents a defect near the center of the viewport, the system dynamically decrements the good count, increments the defective count, and updates the database record in-place.
* **Premium Dark Mode GUI**: A high-DPI scaling PyQt6 interface featuring:
  * Real-time camera feed visualization with color-coded bounding boxes and pixel-perfect segmentation overlays.
  * Custom **Circular Progress Gauges** for real-time quality rates.
  * Multi-metric statistics charts showing total inspected, good, and defective counters.
  * Dynamic Progress Bars representing proportion breakdowns of different defects.
* **SQLite Persistent Logging**: Automatic insertion and updates of inspection results in a local SQLite database (`database/inspection.db`), allowing real-time statistical correction and traceability.
* **Automated Exporters**: Automatically generates high-quality inspection reports when the video stream or camera is stopped:
  * **CSV Reports**: Raw logs containing timestamp, defect type, confidence, and status.
  * **PDF Reports**: Publication-grade summaries containing visual statistics tables, quality scores, and defect breakdown percentages, rendered natively via PyQt6 `QPdfWriter`.
  * **Excel Reports**: Multi-tab formatted logs containing both session KPIs and detailed history.

---

## 📁 Project Structure

```bash
screw-defect-detection-hmi/
├── main.py                  # Main GUI entrypoint (handles High-DPI and stylesheet setup)
├── README.md                # Project documentation
├── Project_Report.md        # Presentation & Engineering Walkthrough report
├── core/                    # Core Python pipeline
│   ├── camera.py            # Video stream source receiver (file, camera, RTSP)
│   ├── detector.py          # YOLO model wrapper, class mapping, mask overlay, and prediction pipeline
│   ├── tracker.py           # Centroid/IoU/Kalman tracking and 2D isotropic detection merging logic
│   ├── database.py          # SQLite database connection, schema setup, and updates
│   ├── statistics.py        # Real-time counter tracker and quality percentage calculator
│   └── exporter.py          # PDF, Excel & CSV reporting engine (built using QPdfWriter and Pandas)
├── ui/                      # PyQt6 User Interface components
│   ├── main_window.py       # Main GUI layout, video slots, and widget callbacks
│   ├── circular_gauge.py    # Custom glassmorphic circular progress widget
│   └── style.py             # Dark mode theme style definitions
├── database/                # SQLite local DB directory
│   └── inspection.db        # Persistent SQL inspection log
├── models/                  # ML Models directory
│   └── best.onnx            # YOLOv8s-seg ONNX format weights
├── reports/                 # Output directory for exported PDF, Excel, and CSV reports
├── snapshots/               # Snapshot images of detected defects
└── videos/                  # Conveyor test video directory
```

---

## 📦 Dataset & Model Weights

The custom dataset used for training this model is hosted on Kaggle. You can download the dataset and view the model training details at the link below:

* **Kaggle Dataset**: [Screw Defect Instance Segmentation Dataset](https://www.kaggle.com/datasets/mahboobxalam/screw-defect-instance-segmentation-dataset)

---

## 🛠️ Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. You can install all required external packages using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Running the Dashboard
To start the dashboard interface:
```bash
python3 main.py
```

### 3. Running with a Custom Video/Camera
The video stream source can be configured directly through the dashboard UI. Select standard input files (such as the default sample `videos/VID_2.mp4` included in the repository), connect live local/RTSP cameras, or input a network mobile camera feed URL.

---

## 📊 Analytics and Reporting
* **Inspection Database**: Logs are persistent and saved in `database/inspection.db`.
* **Exporting Reports**: When the video source or camera is stopped, the system automatically generates a PDF report, an Excel spreadsheet, and a CSV datasheet inside the `reports/` folder.
* **Dynamic Corrections**: If a screw is initially identified as "good" but later presents a defect (e.g., thread defect) before exiting the screen, the system automatically decrements the good count, increments the defective count, updates the local database row using its original database ID, and refreshes the dashboard in real-time.
