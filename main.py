import sys
from pathlib import Path
from PyQt6.QtCore import Qt

# Set High DPI scaling policy before constructing QApplication
# This ensures crisp rendering and prevents layout truncation on screens with fractional scaling (e.g. 125%, 150%)
if sys.platform == "win32":
    try:
        sys.argv.append("--platform")
        sys.argv.append("windows:dpiawareness=2")  # Enable per-monitor DPI awareness on Windows if run there
    except Exception:
        pass

from PyQt6.QtWidgets import QApplication

# PassThrough policy allows fractional scale factors (e.g. 1.5x) instead of rounding them to integer factors (e.g. 1x or 2x)
QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.main_window import MainWindow
from ui.style import DARK_STYLE

app = QApplication(sys.argv)
app.setStyleSheet(DARK_STYLE)

window = MainWindow()
window.show()

sys.exit(app.exec())