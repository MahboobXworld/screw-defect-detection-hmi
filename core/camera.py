import cv2


class Camera:
    """
    Handles video capture from webcam or a video file.

    Usage:
        cam = Camera(source=0)       # 0 = default webcam
        cam = Camera(source="video.mp4")  # video file

        cam.start()
        frame = cam.read()
        cam.stop()
    """

    def __init__(self, source=0):
        self.source = source
        self.cap = None

    def start(self):
        """Open the camera/video source."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source: {self.source}"
            )

    def read(self):
        """
        Read one frame.
        Returns the frame (numpy array) or None if failed.
        """
        if self.cap is None:
            return None

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def stop(self):
        """Release the camera resource."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def is_open(self):
        """Check if camera is currently open."""
        return self.cap is not None and self.cap.isOpened()