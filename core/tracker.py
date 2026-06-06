import numpy as np

def calculate_iou(boxA, boxB):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def clean_detections(detections):
    """
    Applies horizontal grouping/merging of detections belonging to the same screw,
    followed by Non-Maximum Suppression (NMS) and proximity merging to clear overlapping 
    or close boxes.
    """
    if not detections:
        return []

    # 1. Horizontal Grouping & Merging of detections belonging to the same physical screw.
    # Since screws are vertically oriented and move horizontally on the conveyor,
    # detections on the same physical screw will align horizontally (close X centroids or overlap).
    n = len(detections)
    parent = list(range(n))
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            
    for i in range(n):
        det1 = detections[i]
        b1 = det1['bbox']
        cx1 = (b1[0] + b1[2]) / 2.0
        w1 = b1[2] - b1[0]
        
        for j in range(i + 1, n):
            det2 = detections[j]
            b2 = det2['bbox']
            cx2 = (b2[0] + b2[2]) / 2.0
            w2 = b2[2] - b2[0]
            
            dist_x = abs(cx1 - cx2)
            inter_x = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            overlap_x = inter_x / min(w1, w2) if min(w1, w2) > 0 else 0
            
            if dist_x < 50 or overlap_x > 0.5:
                union(i, j)
                
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(detections[i])
        
    merged_detections = []
    for group in groups.values():
        x1 = min(det['bbox'][0] for det in group)
        y1 = min(det['bbox'][1] for det in group)
        x2 = max(det['bbox'][2] for det in group)
        y2 = max(det['bbox'][3] for det in group)
        
        defects = [det for det in group if det['label'] != 'good_screw']
        if defects:
            best_defect = max(defects, key=lambda x: x['conf'])
            label = best_defect['label']
            conf = best_defect['conf']
        else:
            best_good = max(group, key=lambda x: x['conf'])
            label = 'good_screw'
            conf = best_good['conf']
            
        merged_detections.append({
            'bbox': [x1, y1, x2, y2],
            'label': label,
            'conf': conf
        })

    # 2. Standard IoU NMS (IoU > 0.35)
    merged_detections.sort(key=lambda x: x['conf'], reverse=True)
    keep_iou = []
    for det in merged_detections:
        overlap = False
        for kept in keep_iou:
            if calculate_iou(det['bbox'], kept['bbox']) > 0.35:
                overlap = True
                break
        if not overlap:
            keep_iou.append(det)
            
    # 3. Proximity merging (distance between centroids < 50 pixels)
    # We prioritize defects over good_screw during proximity merging
    keep_iou.sort(key=lambda x: (0 if x['label'] == 'good_screw' else 1, x['conf']), reverse=True)
    
    keep_final = []
    for det in keep_iou:
        too_close = False
        det_cx = (det['bbox'][0] + det['bbox'][2]) / 2.0
        det_cy = (det['bbox'][1] + det['bbox'][3]) / 2.0
        for kept in keep_final:
            kept_cx = (kept['bbox'][0] + kept['bbox'][2]) / 2.0
            kept_cy = (kept['bbox'][1] + kept['bbox'][3]) / 2.0
            dist = np.sqrt((det_cx - kept_cx)**2 + (det_cy - kept_cy)**2)
            if dist < 50:
                too_close = True
                break
        if not too_close:
            keep_final.append(det)
            
    return keep_final

class ScrewKalmanFilter:
    def __init__(self, bbox):
        # Initialize state: [cx, cy, w, h, vx, vy]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        
        self.x = np.array([[cx], [cy], [w], [h], [0.0], [0.0]], dtype=np.float32)
        
        # State covariance P
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0]).astype(np.float32)
        
        # State transition matrix F (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix H
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Measurement noise covariance R
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)
        
        # Process noise covariance Q
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.1, 0.1]).astype(np.float32)

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.get_bbox()

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(6, dtype=np.float32) - np.dot(K, self.H), self.P)
        return self.get_bbox()

    def get_bbox(self):
        cx, cy, w, h = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return [float(x1), float(y1), float(x2), float(y2)]

class Track:
    def __init__(self, track_id, bbox, label, conf, start_frame):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.label = label
        self.conf = conf
        self.start_frame = start_frame
        self.last_seen_frame = start_frame
        self.history = [(label, conf)]
        self.counted = False
        self.cx_history = [(bbox[0] + bbox[2]) / 2.0]
        self.kf = ScrewKalmanFilter(bbox)

    def update(self, bbox, label, conf, frame_idx):
        self.bbox = self.kf.update(bbox)
        self.last_seen_frame = frame_idx
        self.history.append((label, conf))
        self.cx_history.append((self.bbox[0] + self.bbox[2]) / 2.0)
        
        # Prioritize defects over good_screw in history labels
        defect_labels = [h[0] for h in self.history if h[0] != "good_screw"]
        if defect_labels:
            self.label = defect_labels[-1]
            self.conf = [h[1] for h in self.history if h[0] == self.label][-1]
        else:
            self.label = "good_screw"
            self.conf = max([h[1] for h in self.history])

    def predict_update(self, bbox):
        self.bbox = bbox
        self.cx_history.append((bbox[0] + bbox[2]) / 2.0)

class IoUTracker:
    def __init__(self, iou_threshold=0.05, max_lost_frames=5):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracks = []
        self.frame_idx = 0
        self.conveyor_direction = None  # "L2R" or "R2L" or None
        self.direction_history = []

    def reset(self):
        """Resets the tracker state."""
        self.next_id = 1
        self.tracks = []
        self.frame_idx = 0
        self.conveyor_direction = None
        self.direction_history = []

    def update(self, detections):
        """
        Updates tracks with new detections using a hybrid IoU and centroid distance matching.
        detections: list of dict, each containing:
            'bbox': [x1, y1, x2, y2]
            'label': str
            'conf': float
        """
        self.frame_idx += 1
        
        # Clean double detections and close duplicate boxes
        detections = clean_detections(detections)
        
        # Filter active tracks (still within the allowed lost-frame window)
        active_tracks = [t for t in self.tracks if self.frame_idx - t.last_seen_frame < self.max_lost_frames]
        
        # Run prediction for active tracks
        predicted_bboxes = {}
        for track in active_tracks:
            predicted_bboxes[track.track_id] = track.kf.predict()
        
        matched_detections = set()
        matched_tracks = set() # indices in active_tracks
        
        # Greedy matching based on IoU and centroid proximity using predicted bboxes
        if active_tracks and detections:
            iou_matrix = np.zeros((len(active_tracks), len(detections)))
            dist_matrix = np.zeros((len(active_tracks), len(detections)))
            for t_idx, track in enumerate(active_tracks):
                pred_bbox = predicted_bboxes[track.track_id]
                track_cx = (pred_bbox[0] + pred_bbox[2]) / 2.0
                track_cy = (pred_bbox[1] + pred_bbox[3]) / 2.0
                for d_idx, det in enumerate(detections):
                    det_cx = (det['bbox'][0] + det['bbox'][2]) / 2.0
                    det_cy = (det['bbox'][1] + det['bbox'][3]) / 2.0
                    
                    iou_matrix[t_idx, d_idx] = calculate_iou(pred_bbox, det['bbox'])
                    dist_matrix[t_idx, d_idx] = np.sqrt((track_cx - det_cx)**2 + (0.1 * (track_cy - det_cy))**2)
            
            # Match candidates Phase 1: IoU >= self.iou_threshold
            matches = []
            for t_idx in range(len(active_tracks)):
                for d_idx in range(len(detections)):
                    iou = iou_matrix[t_idx, d_idx]
                    if iou >= self.iou_threshold:
                        matches.append((iou, t_idx, d_idx))
            
            # Sort matches by highest IoU first
            matches.sort(key=lambda x: x[0], reverse=True)
            
            for iou, t_idx, d_idx in matches:
                if t_idx not in matched_tracks and d_idx not in matched_detections:
                    matched_tracks.add(t_idx)
                    matched_detections.add(d_idx)
                    
                    # Record direction sample
                    track = active_tracks[t_idx]
                    track_cx = track.cx_history[-1]
                    det_cx = (detections[d_idx]['bbox'][0] + detections[d_idx]['bbox'][2]) / 2.0
                    dx = det_cx - track_cx
                    if abs(dx) > 10 and abs(dx) < 180:
                        self.direction_history.append(1 if dx > 0 else -1)
                        if len(self.direction_history) > 10:
                            self.direction_history.pop(0)
                        if len(self.direction_history) >= 5:
                            net_sum = sum(self.direction_history)
                            if net_sum >= 3:
                                self.conveyor_direction = "L2R"
                            elif net_sum <= -3:
                                self.conveyor_direction = "R2L"
                                
                    active_tracks[t_idx].update(
                        detections[d_idx]['bbox'],
                        detections[d_idx]['label'],
                        detections[d_idx]['conf'],
                        self.frame_idx
                    )
            
            # Match candidates Phase 2: Centroid Proximity with Auto-Direction
            proximity_matches = []
            for t_idx in range(len(active_tracks)):
                if t_idx in matched_tracks:
                    continue
                track = active_tracks[t_idx]
                pred_bbox = predicted_bboxes[track.track_id]
                track_cx = (pred_bbox[0] + pred_bbox[2]) / 2.0
                for d_idx in range(len(detections)):
                    if d_idx in matched_detections:
                        continue
                    det = detections[d_idx]
                    det_cx = (det['bbox'][0] + det['bbox'][2]) / 2.0
                    
                    dx = det_cx - track_cx
                    dist = dist_matrix[t_idx, d_idx]
                    
                    valid_direction = True
                    if self.conveyor_direction == "L2R":
                        if dx < -20:
                            valid_direction = False
                    elif self.conveyor_direction == "R2L":
                        if dx > 20:
                            valid_direction = False
                            
                    if valid_direction and dist < 80:
                        proximity_matches.append((dist, t_idx, d_idx, dx))
            
            # Sort proximity matches ascending (closest first)
            proximity_matches.sort(key=lambda x: x[0])
            for dist, t_idx, d_idx, dx in proximity_matches:
                if t_idx not in matched_tracks and d_idx not in matched_detections:
                    matched_tracks.add(t_idx)
                    matched_detections.add(d_idx)
                    
                    # Record direction sample
                    track = active_tracks[t_idx]
                    track_cx = track.cx_history[-1]
                    det_cx = (detections[d_idx]['bbox'][0] + detections[d_idx]['bbox'][2]) / 2.0
                    dx = det_cx - track_cx
                    if abs(dx) > 10 and abs(dx) < 180:
                        self.direction_history.append(1 if dx > 0 else -1)
                        if len(self.direction_history) > 10:
                            self.direction_history.pop(0)
                        if len(self.direction_history) >= 5:
                            net_sum = sum(self.direction_history)
                            if net_sum >= 3:
                                self.conveyor_direction = "L2R"
                            elif net_sum <= -3:
                                self.conveyor_direction = "R2L"
                                
                    active_tracks[t_idx].update(
                        detections[d_idx]['bbox'],
                        detections[d_idx]['label'],
                        detections[d_idx]['conf'],
                        self.frame_idx
                    )
        
        # Update unmatched active tracks with their predicted state
        for t_idx, track in enumerate(active_tracks):
            if t_idx not in matched_tracks:
                pred_bbox = predicted_bboxes[track.track_id]
                track.predict_update(pred_bbox)
        
        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                new_track = Track(
                    self.next_id,
                    det['bbox'],
                    det['label'],
                    det['conf'],
                    self.frame_idx
                )
                self.next_id += 1
                self.tracks.append(new_track)
                
        # Find retired tracks (tracks that were active in the previous frame but have now been lost)
        retired_tracks = [t for t in self.tracks if self.frame_idx - t.last_seen_frame == self.max_lost_frames]
        
        # Clean up very old dead tracks to prevent memory leak
        self.tracks = [t for t in self.tracks if self.frame_idx - t.last_seen_frame < 100]
        
        # Return currently visible tracks, and tracks retired in this frame
        visible_tracks = [t for t in self.tracks if t.last_seen_frame == self.frame_idx]
        return visible_tracks, retired_tracks
