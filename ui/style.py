"""UI stylesheet definitions for the defect inspection HMI."""

DARK_STYLE = """
QWidget {
    background-color: #121212;
    color: #E0E0E0;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 11px;
}

QFrame#panel_frame {
    background-color: #1C1C1C;
    border: 1px solid #2F2F2F;
    border-radius: 10px;
}

QFrame#card_frame {
    background-color: #1F1F1F;
    border: 1px solid #2F2F2F;
    border-radius: 12px;
}

QFrame#alarm_active_frame {
    background-color: #300D18;
    border: 1px solid #FF3B5C;
    border-radius: 10px;
}

QLabel#title_label {
    color: #FFFFFF;
}

QLabel#card_header {
    color: #FFFFFF;
    font-size: 14px;
    font-weight: bold;
}

QLabel#kpi_title {
    color: #A0A0A0;
    font-size: 11px;
}

QLabel#kpi_value,
QLabel#kpi_value_good,
QLabel#kpi_value_defect {
    font-size: 25px;
    font-weight: bold;
}

QLabel#kpi_value_good {
    color: #00FF88;
}

QLabel#kpi_value_defect {
    color: #FF3B5C;
}

QPushButton {
    background-color: #242424;
    color: #ECECEC;
    border: 1px solid #3A3A3A;
    border-radius: 8px;
    padding: 8px 12px;
}

QPushButton:hover {
    background-color: #2E2E2E;
}

QPushButton:pressed {
    background-color: #3A3A3A;
}

QPushButton#btn_success {
    background-color: #007F4F;
    border: 1px solid #00CC77;
    color: #FFFFFF;
}

QPushButton#btn_success:hover {
    background-color: #00995D;
}

QPushButton#btn_danger {
    background-color: #B00020;
    border: 1px solid #FF3B5C;
    color: #FFFFFF;
}

QPushButton#btn_danger:hover {
    background-color: #C80036;
}

QTableWidget {
    background-color: #141414;
    alternate-background-color: #1C1C1C;
    gridline-color: #2E2E2E;
}

QHeaderView::section {
    background-color: #181818;
    color: #E0E0E0;
    border: 1px solid #2F2F2F;
    padding: 4px;
}

QScrollBar:vertical {
    background: #181818;
    width: 12px;
    margin: 16px 0 16px 0;
}

QScrollBar::handle:vertical {
    background: #333333;
    min-height: 20px;
    border-radius: 6px;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    background: none;
}

QProgressBar {
    background-color: #121212;
    border: 1px solid #2F2F2F;
    border-radius: 6px;
    color: #FFFFFF;
    text-align: center;
}
"""
