import sqlite3
import os
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
DB_PATH = os.path.join(DB_DIR, "inspection.db")

def init_db():
    """Initializes the database directory and tables if they do not exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            defect_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT,
            status TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_inspection(defect_type, confidence, image_path, status):
    """
    Inserts a new inspection record.
    Returns the ID of the inserted row.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO inspections (timestamp, defect_type, confidence, image_path, status)
        VALUES (?, ?, ?, ?, ?)
    """, (timestamp, defect_type, confidence, image_path, status))
    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id

def get_history(limit=100, since=None):
    """
    Fetches recent inspection records.
    If `since` is provided, it filters records with timestamp >= since.
    Returns a list of dicts.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if since is not None:
        cursor.execute("""
            SELECT timestamp, defect_type, confidence, image_path, status
            FROM inspections
            WHERE timestamp >= ?
            ORDER BY id DESC
            LIMIT ?
        """, (since, limit))
    else:
        cursor.execute("""
            SELECT timestamp, defect_type, confidence, image_path, status
            FROM inspections
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            "timestamp": r[0],
            "defect_type": r[1],
            "confidence": r[2],
            "image_path": r[3],
            "status": r[4]
        })
    return history

def clear_history():
    """Clears all records in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM inspections")
    conn.commit()
    conn.close()

def update_inspection(row_id, defect_type, confidence, image_path, status):
    """
    Updates an existing inspection record.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE inspections
        SET defect_type = ?, confidence = ?, image_path = ?, status = ?
        WHERE id = ?
    """, (defect_type, confidence, image_path, status, row_id))
    conn.commit()
    conn.close()

