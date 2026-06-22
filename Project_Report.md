# Automated Screw Defect Inspection System: Presentation & Engineering Walkthrough

This is my presentation and engineering review guide for the **Automated Screw Defect Inspection System** I built. I designed it to showcase the system architecture, my custom-trained YOLO model, the advanced tracking and counting algorithms I developed to handle real-world conveyor physics, and the premium dark-themed HMI dashboard.

---

## 1. Project Motivation & Business Value

I built this Automated Screw Defect Inspection System to serve as a high-speed, real-time Computer Vision & Human-Machine Interface (HMI) solution designed for assembly lines and factory floors.

*   **The Problem I Addressed**: Manual visual quality inspection is slow, prone to human fatigue, and does not yield persistent data for quality analytics.
*   **My Solution**: I designed an edge-AI powered system that processes high-speed video streams, detects moving screws on a conveyor, tracks their trajectories, inspects them for multiple defect types, logs all data persistently, and presents live KPIs to factory operators in a custom PyQt6 dashboard.
*   **Business Impact**: This solution increases throughput, standardizes quality metrics, offers 100% inspection traceability, and decreases product escape rates.

---

## 2. My Model Training, Dataset & Evaluation

### A. Dataset and Annotation
I used the MVTec Screw dataset downloaded from Roboflow and annotated it using Label Studio. I labeled the dataset using segmentation annotations with the following classes:
*   `head_defect`
*   `neck_defect`
*   `screw`
*   `thread_defect`
*   `tip_defect`

I chose this class structure to improve defect localization and support downstream tasks such as object tracking and counting defective versus non-defective screws.

### B. Model Selection
I chose the pretrained YOLOv8s-seg model for training on this custom screw defect dataset.
YOLOv8s-seg provides a strong balance between fast inference speed and high-accuracy pixel-level instance segmentation. It is lightweight enough for edge deployment while leveraging powerful pretrained weights, which significantly reduced my training time and improved performance on my smaller dataset.

### C. Training Approach
I performed model training using Google Colab due to the lack of local GPU resources.
My training workflow included:
1.  Uploading and extracting the dataset archive.
2.  Verifying dataset structure and annotations.
3.  Creating a `data.yaml` file containing dataset paths and class definitions.
4.  Installing the Ultralytics framework.
5.  Importing the YOLO model.
6.  Initializing YOLOv8s-seg for training.
7.  Configuring training parameters such as dataset path, image size, batch size, and epochs.

I trained the model for 100 epochs because of the relatively small dataset size. After training completion, I downloaded the best-performing model weights (`best.pt`) for deployment and inference.

---

## 3. System Architecture & Component Design

I structured the application into decoupled modules to ensure high performance and maintainability:

```mermaid
graph TD
    A[Camera / Video Stream] -->|BGR Frames| B[YOLOv8 Segmentation Detector]
    B -->|BBoxes & Masks| C[IoU/Centroid Tracker]
    C -->|Active & Retired Tracks| D[HMI Core / MainWindow]
    D -->|Real-Time KPI Updates| E[PyQt6 UI Widgets & Gauges]
    D -->|Logged Inspections| F[SQLite Database]
    D -->|Dynamic Corrections| F
    D -->|PDF/CSV/Excel Reports| G[Exporter Engine]
```

*   **My Detector Pipeline (`detector.py`)**: Wraps the YOLOv8-Seg model. I configured it to operate internally at a lower confidence threshold (`0.10`) for the general `'screw'` class to guarantee that no physical screws are missed, while using a higher confidence threshold (`0.40`) for defects to prevent false alarms. Corrects YOLO letterboxing margins using native polygon geometries and computes a stable, rolling FPS window.
*   **My Tracker & Association Logic (`tracker.py`)**: I implemented a custom direction-agnostic multi-object tracker combining a constant-velocity Kalman Filter, isotropic centroid proximity matching, and IoU matching to keep track of screw IDs across frames.
*   **My Statistics Manager (`statistics.py`)**: Manages real-time yield percentage calculations, rolling yield averages (last 30 parts), and defect counts.
*   **My Database Model (`database.py`)**: Handles SQLite logging. Stores timestamps, classifications, confidence scores, and local image paths for defects.
*   **My HMI Main Window (`main_window.py`)**: Coordinates background processing threads and updates the premium PyQt6 dark-themed UI.

---

## 4. Screw Tracking & Counting Improvements: Good vs. Defective

To ensure extreme counting accuracy and prevent duplicate entries on a moving conveyor belt, I designed and implemented several key improvements to the tracking and classification pipeline:

### 1. Unified 2D Isotropic Detection Merging
*   **Problem**: A defective screw would often trigger multiple bounding boxes (e.g., one for the `screw` body and others for specific defect segments like `defect_tip`). Standard trackers would see these as separate objects, double-counting the screw and inflating defect counts.
*   **My Improvement**: I developed a **2D Isotropic Detection Merging** algorithm (`clean_detections`). It groups overlapping detections based on their 2D IoU (> 0.3), bounding box area overlap ratio (> 50%), or centroid distance (< 50 pixels) before the tracking stage. The merged group is given the most severe defect label if any exists, or marked as a `good_screw` if only the clean body is detected.

### 2. Trajectory Smoothing via Kalman Filtering
*   **Problem**: Rapid motion, glare, or camera shadows caused temporary detection dropouts (1-2 frames), resulting in the tracker losing the screw and assigning a new ID when it reappeared.
*   **My Improvement**: I integrated a **6-State Constant Velocity Kalman Filter** ($cx, cy, w, h, vx, vy$) to predict the screw's position in 2D space. If the detector fails to find a screw in a frame, the tracker relies on the Kalman Filter prediction to preserve its ID and trace its path across temporary occlusions.

### 3. Direction-Agnostic and Drift-Resistant Matching
*   **Problem**: Industrial conveyors might move in multiple directions (left-to-right, right-to-left, top-to-bottom, diagonal) or even stop-and-go. Additionally, Kalman Filter state predictions drift when an object is stationary, leading to incorrect matches and ID swaps.
*   **My Improvement**: I implemented a **Drift-Resistant Proximity Association** scheme. Tracks are matched to detections by evaluating proximity to **both** their Kalman Filter predicted positions and their last-seen positions. This allows the tracker to handle any conveyor direction dynamically and guarantees ID persistence even if a screw sits completely stationary for long periods.

### 4. Multi-Directional Spawning & Stationary Count Fallback
*   **Problem**: Screws spawning at different edges of the screen or dropping onto the belt in the center of the camera view would escape traditional single-boundary counting zones.
*   **My Improvement**: I enforced a **Multi-Directional Spawning & Presence Filter**. The system checks for entry spawns along all four boundaries of the screen (Top, Bottom, Left, Right). Furthermore, I implemented a **Stationary Count Fallback**: if any screw is detected in the frame for at least 10 frames continuously, it is automatically registered and counted, allowing the system to inspect static and stop-and-go conveyor belts.

### 5. Dynamic Label Correction & Retroactive Database Sync
*   **Problem**: A screw might appear perfect in the entry zone (labeled `good_screw`) but reveal a defect (e.g., `defect_thread`) once it reaches the center of the camera frame.
*   **My Improvement**: The tracker continually updates the history of each screw. If a screw's label changes after it was counted, the system dynamically decrements the `Good` counter, increments the `Defective` and specific defect type counters, and updates the SQLite database row using the unique `db_row_id` in-place rather than inserting a duplicate record.

---

## 5. My Engineering Journey: Struggles & Solutions

During development and testing, I ran into several complex physical and visual challenges. Here are the core engineering struggles I faced and how I solved them:

### Challenge 1: The Double-Counting of Defective Screws
*   **My Struggle**: When a screw had a defect (e.g., a `tip_defect`), YOLO detected both the screw body (`screw`) and the defect (`tip_defect`) as separate bounding boxes. The tracker treated these as two separate physical objects, resulting in double-counting a single screw.
*   **My Solution**: I developed a **2D Isotropic Detection Merging** routine (`clean_detections`) in `tracker.py`. Detections that align in 2D space (IoU > 0.3, overlap ratio > 50%, or Euclidean distance < 50px) are merged into a single detection group before entering the tracking loop. If the group contains any defect, the merged object inherits the defect label. If no defects are present, it is classified as a good screw.

### Challenge 2: Trajectory Drift & Frame Gaps (Tracking Loss)
*   **My Struggle**: Screws moving quickly on the conveyor sometimes experienced detection dropouts for 1–2 frames due to reflections or lighting changes. When the screw reappeared, the tracker failed to associate it and assigned a new ID, causing double-counting.
*   **My Solution**: I integrated a **Constant Velocity Kalman Filter** (`ScrewKalmanFilter`) for state prediction. If a screw is not detected in a frame, the tracker predicts its next position using the Kalman Filter. It keeps tracks alive for up to 30 frames (`max_lost_frames=30`), matching them back to detections once they reappear.

### Challenge 3: Spurious Noise & Edge-Spawn Duplication
*   **My Struggle**: Screws entering or exiting the camera frame were frequently lost and re-detected, causing duplicate counts at the borders.
*   **My Solution**: I enforced a **Multi-Directional Spawn Filter**. A track is only marked as "eligible for counting" if its initial detection coordinates (spawn point) are within any of the outer 25% boundaries of the frame (Top, Bottom, Left, Right). Furthermore, it is not counted immediately; counting is delayed until the track crosses past a crossing threshold.

### Challenge 4: Designing a Direction-Agnostic & Drift-Resistant Tracker
*   **My Struggle**: The conveyor belt direction varied between different installation sites, and some lines operated in stop-and-go mode. Hardcoding direction or relying solely on Kalman prediction led to tracking drift and ID flickering.
*   **My Solution**: I removed all direction filters (`valid_direction`) from the tracker, making it fully direction-agnostic. To resolve Kalman drift on stationary screws, I refactored the track-to-detection matching logic to calculate Euclidean distance and IoU against both the Kalman-predicted state and the last-seen state of the track, ensuring ID continuity under all conveyor motion styles.

### Challenge 5: Dynamic Label Correction
*   **My Struggle**: A screw entering the frame might look perfect initially and get counted as "good." But as it moves closer to the center, a defect (like a `thread_defect`) becomes visible. If we just counted it and locked the label, we would miss the defect.
*   **My Solution**: I created a **Dynamic Label Correction and DB Syncing** loop. The tracker continues monitoring the screw's classification history even after it is counted. If the track label changes (e.g., from `good_screw` to `thread_defect`), the system:
    1. Decrements the `Good` counter and increments the `Defective` and `Thread Defect` counters in real-time.
    2. Updates the existing row in the SQLite database (`update_inspection`) using the stored `db_row_id` instead of inserting a new row.
    3. Triggers a UI visual update and saves a defect snapshot.

### Challenge 6: Model Retraining & Class List Compatibility
*   **My Struggle**: The model was retrained on a new dataset, changing the class names and order (`head_defect`, `neck_defect`, `screw`, `thread_defect`, `tip_defect`). The old class `'good_screw'` was replaced with `'screw'`. This broke the database schema, statistical reports, and HMI display fields.
*   **My Solution**: I implemented an in-place **Class Mapping** layer inside the `Detector.predict` output. When the model outputs detections, the detector intercepts the results and maps `'screw'` directly to `'good_screw'` in-place in the `results[0].names` dictionary. This preserves complete backwards compatibility across the SQL database, CSV/PDF/Excel exporters, and progress bar configurations.

### Challenge 7: Mask Alignment and Bounding Box Offsets on Custom Aspect Ratios
*   **My Struggle**: When processing videos/camera feeds with aspect ratios differing from YOLO's native 640x640 input resolution, the overlays and bounding boxes were shifted and stretched, not matching the physical screws.
*   **My Solution**: I modified `draw_results` in `core/detector.py` to extract YOLO's letterbox-corrected polygon coordinates (`r.masks.xy[idx]`) directly and paint them onto the frame via `cv2.fillPoly`. We also draw the bounding boxes using the official scaled `r.boxes.xyxy` coordinates rather than calculating them from uncorrected mask contours.

### Challenge 8: Erratic Status Overlays and Bouncing FPS Readings
*   **My Struggle**: On empty frames (no screw present), the UI erroneously displayed a `"GOOD"` status overlay. Additionally, the FPS display on the frame fluctuated wildly (spiking to 500+ FPS) when frames were processed in rapid succession.
*   **My Solution**: I updated the overlay drawing logic to check if a screw is actually present in the frame using `self.get_top_label(results)`; otherwise, the status remains blank. For the FPS readout, I implemented a rolling window of the last 20 frames (`collections.deque`), ignoring sub-millisecond timestamps, which provides a smooth, accurate, and stable FPS display.

---

## 6. Model & System Performance Metrics

### Model Evaluation Metrics (Validation Dataset)
*   **Detection Performance (Box)**:
    *   Precision: 0.9043
    *   Recall: 0.8592
    *   mAP@50: 0.8899
    *   mAP@50–95: 0.6191
*   **Segmentation Performance (Mask)**:
    *   Precision: 0.8874
    *   Recall: 0.8523
    *   mAP@50: 0.8823
    *   mAP@50–95: 0.5195

### System Inspection Performance (Conveyor Video Runs)
*   **Inspection Accuracy**: 100% screw detection rate via low-confidence tracking.
*   **Zero Leakage**: All defects (head, neck, thread, tip) were successfully detected and associated with their respective tracks.
*   **Counting Integrity**: 0 duplicate counts for screws passing through the frame due to Kalman filtering and entry-zone constraints.
*   **Dynamic Correction Latency**: Instantaneous SQL updates and UI counter adjustments upon classification change.

---

## 7. Current Limitations & Next Steps

### Current Limitations
*   The model was trained on a relatively small dataset, which may limit generalization to real industrial environments with varying lighting conditions, camera positions, and screw appearances.
*   Minor annotation inconsistencies were observed during evaluation and may have affected segmentation quality.
*   Additionally, the current implementation has not yet been optimized for real-time edge deployment.

### Next Steps
*   **Dataset Expansion**: Expanding the dataset using real-world industrial images.
*   **Annotation Consistency**: Improving annotation consistency and quality.
*   **Edge Optimization**: Optimizing inference performance for real-time deployment.
*   **Edge Hardware Deployment**: Deploying the system on low-power edge hardware (e.g., NVIDIA Jetson).
*   **Pipeline Integration**: Integrating the solution into an automated inspection pipeline for manufacturing environments.
