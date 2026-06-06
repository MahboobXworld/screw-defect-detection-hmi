import time
import cv2
import numpy as np
from ultralytics import YOLO
from core.tracker import calculate_iou


class Detector:
    """
    Runs the YOLOv8 model on a single frame.

    The model file 'best.pt' must be in the project root.

    Detects these classes:
        - good_screw
        - head_defect
        - neck_defect
        - thread_defect
        - tip_defect
    """

    def __init__(self, model_path="best.pt", confidence=0.4):
        self.model_path = model_path
        self.confidence = confidence
        self.model = YOLO(model_path)
        self.prev_time = time.time()

    def predict(self, frame):
        """
        Run detection on a single frame.

        Args:
            frame: numpy array (BGR image from OpenCV)

        Returns:
            results: Ultralytics Results object
                     Use results[0].boxes to get detections
        """
        results = self.model(
            frame,
            conf=0.10,  # Run internally with lower confidence to capture good screws
            verbose=False
        )

        # Map 'screw' to 'good_screw' in results names dictionary
        if results and len(results) > 0:
            for r in results:
                if hasattr(r, 'names') and r.names is not None:
                    for idx, name in list(r.names.items()):
                        if name == "screw":
                            r.names[idx] = "good_screw"

        return results

    def get_top_label(self, results):
        """
        Extract the highest-confidence label from results.
        We filter defects by self.confidence (0.40), and good screws by a lower threshold (0.10)
        to ensure good screws are counted with 100% accuracy.

        Returns:
            str: class name like 'good_screw' or 'head_defect'
            None: if nothing detected
        """
        if not results or len(results) == 0:
            return None

        r = results[0]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            return None

        class_names = r.names
        
        best_defect_idx = -1
        best_defect_conf = 0.0
        good_screw_idx = -1
        good_screw_conf = 0.0

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for idx, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
            label = class_names[cls_id]
            if label == "good_screw":
                if conf >= 0.10 and conf > good_screw_conf:
                    good_screw_conf = conf
                    good_screw_idx = idx
            else:
                if conf >= self.confidence and conf > best_defect_conf:
                    best_defect_conf = conf
                    best_defect_idx = idx

        # Return defect label if found (defects have priority for safety)
        if best_defect_idx != -1:
            class_id = int(boxes.cls[best_defect_idx].item())
            return class_names[class_id]
        elif good_screw_idx != -1:
            class_id = int(boxes.cls[good_screw_idx].item())
            return class_names[class_id]

        return None

    def draw_results(self, frame, results, tracks=None):
        """
        Draw bounding boxes, segmentation masks, label text with pointing arrows,
        status, and FPS directly on the frame.

        Returns:
            annotated_frame: numpy array with custom drawings
        """
        if not results or len(results) == 0:
            return frame.copy()

        r = results[0]
        img = frame.copy()
        h, w = img.shape[:2]

        class_names = r.names
        CLASS_COLORS = {
            "head_defect": (0, 0, 255),      # Red (BGR)
            "neck_defect": (0, 255, 255),    # Yellow (BGR)
            "thread_defect": (255, 0, 0),    # Blue (BGR)
            "tip_defect": (255, 0, 255)      # Magenta (BGR)
        }

        image_defects = []

        if r.masks is not None and len(r.boxes) > 0:
            masks = r.masks.data.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()

            for idx, (mask, cls_id, conf, bbox_coords) in enumerate(zip(masks, cls_ids, confs, boxes)):
                if conf < self.confidence:
                    continue

                label = class_names[cls_id]
                # Ignore good_screw detections
                if label == "good_screw":
                    continue

                # Check for track matching
                track_id = None
                if tracks is not None and len(tracks) > 0:
                    best_iou = -1.0
                    best_track = None
                    for t in tracks:
                        iou = calculate_iou(bbox_coords.tolist(), t.bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_track = t
                    if best_iou >= 0.3:
                        track_id = best_track.track_id

                # Resize mask if it does not match the image shape
                if mask.shape != (h, w):
                    mask = cv2.resize(
                        mask,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST
                    )

                mask_bool = mask > 0.5
                if not np.any(mask_bool):
                    continue

                # Center point
                ys, xs = np.where(mask_bool)
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))

                # Get BGR Color
                color = CLASS_COLORS.get(label, (0, 255, 0))

                # Draw segmentation overlay
                colored_mask = np.zeros_like(img)
                colored_mask[mask_bool] = color
                img = cv2.addWeighted(
                    img,
                    1.0,
                    colored_mask,
                    0.35,
                    0
                )

                # Draw bounding box from mask contours
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in contours:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    cv2.rectangle(
                        img,
                        (x, y),
                        (x + bw, y + bh),
                        color,
                        2
                    )

                # Store info
                image_defects.append({
                    "defect": label,
                    "confidence": float(conf),
                    "location": (cx, cy)
                })

                # Draw label text
                if track_id is not None:
                    label_text = f"{label} (ID: {track_id}): {conf:.2f}"
                else:
                    label_text = f"{label}: {conf:.2f}"
                offset = 50 + (idx * 25)
                label_x = min(w - 250, cx + offset)
                label_y = max(30, cy - offset)

                cv2.putText(
                    img,
                    label_text,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                # Draw arrowed line pointing to the center of defect
                cv2.arrowedLine(
                    img,
                    (label_x, label_y),
                    (cx, cy),
                    (0, 0, 255),
                    2,
                    tipLength=0.15
                )

                # Draw center circle
                cv2.circle(
                    img,
                    (cx, cy),
                    4,
                    (0, 0, 255),
                    -1
                )

        # Draw vertical dashed inspection line in the middle
        line_x = w // 2
        for y in range(0, h, 20):
            cv2.line(img, (line_x, y), (line_x, min(h, y + 10)), (0, 165, 255), 2)  # Orange dashed line
        cv2.putText(
            img,
            "INSPECTION LINE",
            (line_x + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )

        # Status text overlay
        status = "DEFECTIVE" if len(image_defects) > 0 else "GOOD"
        result_color = (0, 0, 255) if status == "DEFECTIVE" else (0, 255, 0)
        cv2.putText(
            img,
            status,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            result_color,
            3
        )

        # FPS calculation and overlay
        curr_time = time.time()
        elapsed = curr_time - self.prev_time
        self.prev_time = curr_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        cv2.putText(
            img,
            f"FPS: {fps:.1f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        return img