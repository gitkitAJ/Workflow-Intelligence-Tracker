"""SQLite persistence layer for Workflow Intelligence Tracker."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
import pandas as pd

DB_PATH = Path(__file__).with_name("workflow_tracker.db")

def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db(db_path: str | Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_name TEXT NOT NULL,
            description TEXT DEFAULT '',
            category TEXT DEFAULT '',
            status TEXT DEFAULT 'Todo',
            estimated_minutes REAL,
            actual_minutes REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            inputs TEXT DEFAULT '',
            outputs TEXT DEFAULT '',
            workflow_name TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS pair_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_a_id INTEGER NOT NULL,
            task_b_id INTEGER NOT NULL,
            similarity REAL NOT NULL,
            label TEXT NOT NULL,
            notes TEXT DEFAULT '',
            reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(task_a_id, task_b_id),
            FOREIGN KEY(task_a_id) REFERENCES tasks(id) ON DELETE CASCADE,
            FOREIGN KEY(task_b_id) REFERENCES tasks(id) ON DELETE CASCADE
        );
        """)

def add_task(task: dict[str, Any], db_path: str | Path | None = None) -> int:
    fields = ["task_name", "description", "category", "status", "estimated_minutes",
              "actual_minutes", "created_at", "completed_at", "inputs", "outputs", "workflow_name"]
    values = [task.get(k) for k in fields]
    with connect(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO tasks ({','.join(fields)}) VALUES ({','.join(['?']*len(fields))})",
            values,
        )
        return int(cur.lastrowid)

def add_tasks_bulk(df: pd.DataFrame, db_path: str | Path | None = None) -> int:
    count = 0
    for _, row in df.iterrows():
        task = {k: row.get(k) for k in [
            "task_name", "description", "category", "status", "estimated_minutes",
            "actual_minutes", "created_at", "completed_at", "inputs", "outputs", "workflow_name"
        ]}
        task = {k: (None if pd.isna(v) else v) for k, v in task.items()}
        if not task.get("task_name") or not str(task["task_name"]).strip():
            continue
        add_task(task, db_path)
        count += 1
    return count

def get_tasks(db_path: str | Path | None = None) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM tasks ORDER BY id DESC", conn)

def update_task(task_id: int, task: dict[str, Any], db_path: str | Path | None = None) -> None:
    fields = ["task_name", "description", "category", "status", "estimated_minutes",
              "actual_minutes", "completed_at", "inputs", "outputs", "workflow_name"]
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET " + ", ".join(f"{f}=?" for f in fields) + " WHERE id=?",
            [task.get(f) for f in fields] + [task_id],
        )

def delete_task(task_id: int, db_path: str | Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))

def save_review(task_a_id: int, task_b_id: int, similarity: float, label: str,
                notes: str = "", db_path: str | Path | None = None) -> None:
    a, b = sorted((int(task_a_id), int(task_b_id)))
    with connect(db_path) as conn:
        conn.execute("""
            INSERT INTO pair_reviews(task_a_id, task_b_id, similarity, label, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(task_a_id, task_b_id) DO UPDATE SET
              similarity=excluded.similarity, label=excluded.label, notes=excluded.notes,
              reviewed_at=CURRENT_TIMESTAMP
        """, (a, b, float(similarity), label, notes))

def get_reviews(db_path: str | Path | None = None) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query("""
          SELECT r.*, a.task_name AS task_a_name, a.description AS task_a_description,
                 b.task_name AS task_b_name, b.description AS task_b_description
          FROM pair_reviews r
          JOIN tasks a ON a.id=r.task_a_id
          JOIN tasks b ON b.id=r.task_b_id
          ORDER BY r.reviewed_at DESC
        """, conn)

def export_reviews(db_path: str | Path | None = None) -> pd.DataFrame:
    return get_reviews(db_path)
