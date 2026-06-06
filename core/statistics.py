class Statistics:
    """
    Tracks inspection counts during a session.

    Attributes:
        total     : total screws inspected
        good      : screws with no defect
        defective : screws with any defect
        head      : screws with head defect
        neck      : screws with neck defect
        thread    : screws with thread defect
        tip       : screws with tip defect
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters to zero."""
        self.total = 0
        self.good = 0
        self.defective = 0

        self.head = 0
        self.neck = 0
        self.thread = 0
        self.tip = 0
        
        self.recent_results = [] # list of [track_id, defect_type]

    def update(self, track_id, defect_type):
        """
        Update counters based on detection result.
        """
        self.total += 1

        if defect_type == "good_screw":
            self.good += 1
        else:
            self.defective += 1
            if defect_type == "head_defect":
                self.head += 1
            elif defect_type == "neck_defect":
                self.neck += 1
            elif defect_type == "thread_defect":
                self.thread += 1
            elif defect_type == "tip_defect":
                self.tip += 1

        # Track in rolling history for real-time quality
        self.recent_results.append([track_id, defect_type])
        if len(self.recent_results) > 30:
            self.recent_results.pop(0)

    def update_label_change(self, track_id, old_label, new_label):
        """
        Updates counts when a track's label changes dynamically.
        """
        if old_label == new_label:
            return

        # Decrement old counters
        if old_label == "good_screw":
            self.good -= 1
        else:
            self.defective -= 1
            if old_label == "head_defect":
                self.head -= 1
            elif old_label == "neck_defect":
                self.neck -= 1
            elif old_label == "thread_defect":
                self.thread -= 1
            elif old_label == "tip_defect":
                self.tip -= 1

        # Increment new counters
        if new_label == "good_screw":
            self.good += 1
        else:
            self.defective += 1
            if new_label == "head_defect":
                self.head += 1
            elif new_label == "neck_defect":
                self.neck += 1
            elif new_label == "thread_defect":
                self.thread += 1
            elif new_label == "tip_defect":
                self.tip += 1

        # Update rolling history
        for item in self.recent_results:
            if item[0] == track_id:
                item[1] = new_label
                break

    def yield_percent(self):
        """
        Calculate yield percentage.
        Yield = (good / total) * 100

        Returns 100.0 if no screws inspected yet.
        """
        if self.total == 0:
            return 100.0
        return round((self.good / self.total) * 100, 1)

    def rolling_yield_percent(self):
        """
        Calculate rolling yield percentage of the last 30 screws.
        """
        if not self.recent_results:
            return 100.0
        good_count = sum(1 for item in self.recent_results if item[1] == "good_screw")
        return round((good_count / len(self.recent_results)) * 100, 1)


    def most_common_defect(self):
        """
        Return the defect type with the highest count.
        Returns None if no defects recorded.
        """
        defects = {
            "head_defect":   self.head,
            "neck_defect":   self.neck,
            "thread_defect": self.thread,
            "tip_defect":    self.tip,
        }

        if max(defects.values()) == 0:
            return None

        return max(defects, key=defects.get)

    def to_dict(self):
        """Return all stats as a plain dictionary."""
        return {
            "Total Inspected": self.total,
            "Good Screws":     self.good,
            "Total Defective": self.defective,
            "Head Defects":    self.head,
            "Neck Defects":    self.neck,
            "Thread Defects":  self.thread,
            "Tip Defects":     self.tip,
            "Quality":         self.yield_percent(),
        }