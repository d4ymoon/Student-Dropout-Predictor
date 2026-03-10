from pathlib import Path
import sqlite3

# DB path
DB_PATH = Path(__file__).resolve().parent / "student_dropout.db"

def init_db():
    """Initialize DB and create predictions table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                features_json TEXT NOT NULL,
                prediction REAL NOT NULL
            )
        """)
        conn.commit()

def insert_prediction(created_at: str, features_json: str, prediction: float):
    """Insert a prediction record."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, features_json, prediction) VALUES (?, ?, ?)",
            (created_at, features_json, prediction),
        )
        conn.commit()

def fetch_latest(limit: int = 20):
    """Fetch the latest predictions."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT id, created_at, features_json, prediction FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()
    
def fetch_all_predictions():
    """Fetch all predictions from the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT id, created_at, features_json, prediction FROM predictions ORDER BY id DESC"
        )
        return cur.fetchall()