import os
from datetime import datetime
import pandas as pd
from PyQt6.QtGui import QPdfWriter, QTextDocument, QPageSize
from PyQt6.QtCore import QSizeF

# Make sure the reports folder exists
os.makedirs("reports", exist_ok=True)

def build_report_summary(stats, history_list=None):
    """Build a summary dictionary using the live stats values only."""
    return stats.to_dict()


def export_csv(stats, history_list=None):
    """
    Export current session statistics and history to a CSV file.
    Saves to: reports/report_YYYY-MM-DD_HH-MM-SS.csv
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"reports/report_{timestamp}.csv"

    report_summary = build_report_summary(stats)

    # We will write summary stats first, then the history log.
    with open(filename, "w", encoding="utf-8") as f:
        # Title and Timestamp
        f.write("SCREW DEFECT INSPECTION SYSTEM - SESSION SUMMARY\n")
        f.write(f"Exported At:,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary KPI
        f.write("KPI METRICS\n")
        for k, v in report_summary.items():
            if isinstance(v, float):
                f.write(f"{k},{v:.1f}%\n" if k == "Quality" else f"{k},{v}\n")
            else:
                f.write(f"{k},{v}\n")
        f.write("\n")
        
        # History
        if history_list:
            f.write("INSPECTION HISTORY LOG\n")
            f.write("Timestamp,Defect Type,Confidence,Image Path,Status\n")
            for record in history_list:
                img_path = record.get("image_path") or ""
                # escape commas in paths just in case
                img_path = img_path.replace(",", ";")
                f.write(f"{record['timestamp']},{record['defect_type']},{record['confidence']:.4f},{img_path},{record['status']}\n")
                
    print(f"[Exporter] CSV saved: {filename}")
    return filename

def export_excel(stats, history_list=None):
    """
    Export current session statistics and history to an Excel file with multiple tabs.
    Saves to: reports/report_YYYY-MM-DD_HH-MM-SS.xlsx
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"reports/report_{timestamp}.xlsx"

    # Create summary DataFrame
    s_dict = stats.to_dict()
    df_summary = pd.DataFrame(list(s_dict.items()), columns=["Metric", "Value"])
    
    # Create history DataFrame
    if history_list:
        df_history = pd.DataFrame(history_list)
    else:
        df_history = pd.DataFrame(columns=["timestamp", "defect_type", "confidence", "image_path", "status"])

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary KPIs", index=False)
        df_history.to_excel(writer, sheet_name="Inspection Log", index=False)

    print(f"[Exporter] Excel saved: {filename}")
    return filename

def export_pdf(stats, history_list=None):
    """
    Export current session statistics and defect history to a PDF report.
    Uses PyQt6 QPdfWriter with HTML formatting for offline and dependencies-free generation.
    Saves to: reports/report_YYYY-MM-DD_HH-MM-SS.pdf
    """
    timestamp_file = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"reports/report_{timestamp_file}.pdf"
    
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_summary = build_report_summary(stats)
    
    total_inspected = report_summary.get("Total Inspected", stats.total)
    total_defective = report_summary.get("Total Defective", stats.defective)
    good_screws = report_summary.get("Good Screws", stats.good)
    quality_pct = report_summary.get("Quality", round(stats.yield_percent(), 1))
    head_count = report_summary.get("Head Defects", stats.head)
    neck_count = report_summary.get("Neck Defects", stats.neck)
    thread_count = report_summary.get("Thread Defects", stats.thread)
    tip_count = report_summary.get("Tip Defects", stats.tip)
    
    good_pct = round((good_screws / total_inspected * 100) if total_inspected > 0 else 0, 1)
    defect_pct = round((total_defective / total_inspected * 100) if total_inspected > 0 else 0, 1)
    
    # Format Defect list for HTML table
    history_rows = ""
    if history_list:
        # Limit to last 50 entries to keep PDF clean and bounded
        defect_only_history = [r for r in history_list if r["status"] == "DEFECT"]
        for r in defect_only_history[:50]:
            img_name = os.path.basename(r.get("image_path") or "") or "N/A"
            history_rows += f"""
            <tr>
                <td>{r['timestamp']}</td>
                <td><strong style='color:#FF3B5C;'>{r['defect_type'].replace('_', ' ').upper()}</strong></td>
                <td>{r['confidence']:.2f}</td>
                <td>{img_name}</td>
            </tr>
            """
    
    if not history_rows:
        history_rows = "<tr><td colspan='4' style='text-align:center;'>No defects recorded during this run.</td></tr>"

    html_content = f"""
    <html>
    <head>
    <style>
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            margin: 20px;
            color: #333333;
            font-size: 11pt;
            line-height: 1.4;
        }}
        .header {{
            border-bottom: 3px solid #00BFFF;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .title {{
            font-size: 24pt;
            font-weight: bold;
            color: #121212;
            margin: 0;
        }}
        .subtitle {{
            font-size: 10pt;
            color: #777777;
            margin: 5px 0 0 0;
            text-transform: uppercase;
        }}
        .meta-table {{
            width: 100%;
            margin-bottom: 20px;
        }}
        .meta-table td {{
            padding: 3px 0;
        }}
        .kpi-container {{
            width: 100%;
            margin-bottom: 20px;
            clear: both;
        }}
        .kpi-card {{
            display: inline-block;
            width: 30%;
            border: 1px solid #E0E0E0;
            border-radius: 5px;
            padding: 10px;
            margin-right: 2%;
            background-color: #F8F9FA;
            text-align: center;
        }}
        .kpi-title {{
            font-size: 9pt;
            font-weight: bold;
            color: #666666;
            text-transform: uppercase;
        }}
        .kpi-value {{
            font-size: 18pt;
            font-weight: bold;
            color: #00BFFF;
            margin: 5px 0;
        }}
        .kpi-value.success {{
            color: #2ECC71;
        }}
        .kpi-value.danger {{
            color: #E74C3C;
        }}
        .section-title {{
            font-size: 14pt;
            font-weight: bold;
            color: #1E1E1E;
            border-bottom: 1px solid #E0E0E0;
            padding-bottom: 5px;
            margin-top: 25px;
            margin-bottom: 10px;
        }}
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        table.data-table th, table.data-table td {{
            border: 1px solid #E0E0E0;
            padding: 8px 10px;
            text-align: left;
        }}
        table.data-table th {{
            background-color: #F1F3F5;
            color: #495057;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 40px;
            border-top: 1px solid #E0E0E0;
            padding-top: 10px;
            font-size: 8pt;
            color: #999999;
            text-align: center;
        }}
    </style>
    </head>
    <body>
        <div class="header">
            <h1 class="title">SCREW INPSECTION HMI REPORT</h1>
            <div class="subtitle">Production Line Defect Verification Report</div>
        </div>

        <table class="meta-table">
            <tr>
                <td><strong>Report Timestamp:</strong> {formatted_time}</td>
                <td style="text-align: right;"><strong>System Model:</strong> YOLOv8-Seg (v1.2)</td>
            </tr>
            <tr>
                <td><strong>Factory Environment:</strong> Screw Inspection Station 1</td>
                <td style="text-align: right;"><strong>HMI Version:</strong> v1.0.0</td>
            </tr>
        </table>

        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-title">Total Inspected</div>
                <div class="kpi-value">{total_inspected}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Quality</div>
                <div class="kpi-value success">{quality_pct}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Defective Screws</div>
                <div class="kpi-value danger">{total_defective}</div>
            </div>
        </div>

        <div class="section-title">Session Summary Stats</div>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Classification Category</th>
                    <th>Count</th>
                    <th>Proportion of Total</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Good Screws</td>
                    <td>{good_screws}</td>
                    <td>{good_pct}%</td>
                </tr>
                <tr>
                    <td>Defective Screws (Total)</td>
                    <td>{total_defective}</td>
                    <td>{defect_pct}%</td>
                </tr>
                <tr>
                    <td>Quality</td>
                    <td>{quality_pct}%</td>
                    <td></td>
                </tr>
                <tr>
                    <td style="padding-left: 20px;">• Head Defect</td>
                    <td>{head_count}</td>
                    <td>{round((head_count/total_inspected*100) if total_inspected > 0 else 0, 1)}%</td>
                </tr>
                <tr>
                    <td style="padding-left: 20px;">• Neck Defect</td>
                    <td>{neck_count}</td>
                    <td>{round((neck_count/total_inspected*100) if total_inspected > 0 else 0, 1)}%</td>
                </tr>
                <tr>
                    <td style="padding-left: 20px;">• Thread Defect</td>
                    <td>{thread_count}</td>
                    <td>{round((thread_count/total_inspected*100) if total_inspected > 0 else 0, 1)}%</td>
                </tr>
                <tr>
                    <td style="padding-left: 20px;">• Tip Defect</td>
                    <td>{tip_count}</td>
                    <td>{round((tip_count/total_inspected*100) if total_inspected > 0 else 0, 1)}%</td>
                </tr>
            </tbody>
        </table>

        <div class="section-title">Defect History Log (Last 50 Records)</div>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Defect Classification</th>
                    <th>Confidence</th>
                    <th>Snapshot File</th>
                </tr>
            </thead>
            <tbody>
                {history_rows}
            </tbody>
        </table>

        <div class="footer">
            Generated automatically by Screw Defect Inspection HMI System.<br/>
            All data logged securely in SQLite database.
        </div>
    </body>
    </html>
    """

    doc = QTextDocument()
    doc.setHtml(html_content)

    # Print document to QPdfWriter
    writer = QPdfWriter(filename)
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    # 72 points per inch standard A4 layout mapping
    writer.setResolution(150) 
    doc.print(writer)

    print(f"[Exporter] PDF saved: {filename}")
    return filename