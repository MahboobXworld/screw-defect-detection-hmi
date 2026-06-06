from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QRectF

class CircularGauge(QWidget):
    """
    A custom PyQt6 QWidget that renders a modern circular progress gauge.
    Highly suitable for Industrial HMI dashboards to display Yield, OEE, Quality, etc.
    """
    def __init__(self, parent=None, value=100.0, label="Gauge", color=QColor("#00FF88"), suffix="%"):
        super().__init__(parent)
        self.value = value
        self.label = label
        self.color = color
        self.suffix = suffix
        self.bg_color = QColor("#3A3A3A")
        self.setMinimumSize(100, 100)
        
        # Cache fonts and pens
        self.font_value = None
        self.font_label = None
        self.bg_pen = None
        self.fg_pen = None
        self.last_value = -1
        self.last_width = -1
        self.last_height = -1

    def setValue(self, val):
        """Update value (clamped between 0.0 and 100.0) and trigger repaint."""
        new_val = max(0.0, min(100.0, float(val)))
        if new_val != self.value:
            self.value = new_val
            self.update()

    def setLabel(self, text):
        """Update gauge label and trigger repaint."""
        if text != self.label:
            self.label = text
            self.update()

    def setColor(self, color):
        """Update gauge foreground arc color and trigger repaint."""
        self.color = color
        self.fg_pen = None  # Invalidate cache
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        
        # Calculate geometry to keep it centered and circular
        size = min(width, height) - 12
        x = (width - size) / 2
        y = (height - size) / 2
        rect = QRectF(x, y, size, size)

        pen_width = int(size * 0.08)
        pen_width = max(4, min(16, pen_width))

        # Cache invalidation check
        if self.last_width != width or self.last_height != height:
            self.font_value = None
            self.font_label = None
            self.bg_pen = None
            self.fg_pen = None
            self.last_width = width
            self.last_height = height

        # 1. Paint background track arc
        if self.bg_pen is None:
            self.bg_pen = QPen(self.bg_color)
            self.bg_pen.setWidth(pen_width)
            self.bg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(self.bg_pen)
        painter.drawArc(rect, 90 * 16, -360 * 16)

        # 2. Paint foreground active value arc
        if self.fg_pen is None:
            self.fg_pen = QPen(self.color)
            self.fg_pen.setWidth(pen_width)
            self.fg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(self.fg_pen)
        
        span_angle = -int(self.value * 3.6 * 16)
        painter.drawArc(rect, 90 * 16, span_angle)

        # 3. Paint Text Value in the center
        if self.font_value is None:
            font_size_val = max(11, int(size * 0.16))
            self.font_value = QFont("Segoe UI", font_size_val, QFont.Weight.Bold)
        painter.setFont(self.font_value)
        painter.setPen(QColor("#FFFFFF"))
        value_str = f"{self.value:.1f}{self.suffix}"
        
        rect_val = QRectF(x, y + size * 0.18, size, size * 0.35)
        painter.drawText(rect_val, Qt.AlignmentFlag.AlignCenter, value_str)

        # Label text
        if self.font_label is None:
            font_size_lbl = max(8, int(size * 0.085))
            self.font_label = QFont("Segoe UI", font_size_lbl, QFont.Weight.DemiBold)
        painter.setFont(self.font_label)
        painter.setPen(QColor("#A0A0A0"))
        
        rect_lbl = QRectF(x, y + size * 0.54, size, size * 0.28)
        painter.drawText(rect_lbl, Qt.AlignmentFlag.AlignCenter, self.label.upper())
