import sqlite3
from datetime import datetime

DB_NAME = "fog_detection.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT,
        filename TEXT,
        fog_level TEXT,
        confidence REAL,
        warning_level TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_record(filename, fog_level, confidence, warning):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO records(time,filename,fog_level,confidence,warning_level)
    VALUES (?,?,?,?,?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          filename, fog_level, confidence, warning))

    conn.commit()
    conn.close()


def query_records():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM records ORDER BY time DESC")
    rows = cursor.fetchall()

    conn.close()
    return rows