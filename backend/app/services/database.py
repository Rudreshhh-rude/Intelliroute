import os
import sqlite3
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel

logger = logging.getLogger("intelliroute.database")

# Resolve database file path relative to project backend directory
DB_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../database.db"))

def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database file."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Enables column access by name
    return conn

def init_db(default_admin_key: str):
    """Initializes database tables and seeds a default Admin API key if empty."""
    logger.info(f"Initializing SQLite database at: {DB_FILE}")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. API Keys Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS api_keys (
        key TEXT PRIMARY KEY,
        role TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at REAL NOT NULL
    )
    """)
    
    # 2. Observability Metrics Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS route_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        classified_complexity TEXT NOT NULL,
        chosen_model TEXT NOT NULL,
        reasoning TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        completion_tokens INTEGER NOT NULL,
        latency_sec REAL NOT NULL,
        cost_usd REAL NOT NULL,
        timestamp REAL NOT NULL
    )
    """)
    
    # 3. Chat Messages Table (Conversation Memory)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp REAL NOT NULL
    )
    """)
    
    # 4. Running Summaries Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS running_summaries (
        session_id TEXT PRIMARY KEY,
        summary TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """)
    
    conn.commit()
    
    # Seed default admin key if table is completely empty
    cursor.execute("SELECT COUNT(*) FROM api_keys")
    count = cursor.fetchone()[0]
    if count == 0 and default_admin_key:
        logger.info(f"Seeding default Admin API key: {default_admin_key[:8]}...")
        cursor.execute(
            "INSERT INTO api_keys (key, role, status, created_at) VALUES (?, ?, ?, ?)",
            (default_admin_key, "Admin", "active", time.time())
        )
        conn.commit()
        
    conn.close()

# API Key Management

def get_api_key(key: str) -> Optional[Dict[str, Any]]:
    """Retrieves an API key configuration from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM api_keys WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def add_api_key(key: str, role: str, status: str = "active") -> bool:
    """Adds a new API key configuration to database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO api_keys (key, role, status, created_at) VALUES (?, ?, ?, ?)",
            (key, role, status, time.time())
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def revoke_api_key(key: str) -> bool:
    """Revokes an API key by setting status to 'revoked'."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE api_keys SET status = 'revoked' WHERE key = ?", (key,))
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated

# Metrics Logger

def log_route_metrics(
    query: str,
    classified_complexity: str,
    chosen_model: str,
    reasoning: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_sec: float,
    cost_usd: float
):
    """Inserts a new transaction metrics entry into database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO route_metrics (
            query, classified_complexity, chosen_model, reasoning, 
            prompt_tokens, completion_tokens, latency_sec, cost_usd, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            query, classified_complexity, chosen_model, reasoning,
            prompt_tokens, completion_tokens, latency_sec, cost_usd, time.time()
        )
    )
    conn.commit()
    conn.close()

def get_metrics_summary() -> Dict[str, Any]:
    """Computes session overview stats directly from SQL aggregates."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM route_metrics")
    total_queries = cursor.fetchone()[0]
    
    if total_queries == 0:
        conn.close()
        return {
            "total_queries": 0,
            "total_cost": 0.0,
            "avg_latency": 0.0,
            "total_tokens": 0,
            "simple_count": 0,
            "complex_count": 0,
            "model_distribution": {}
        }
        
    cursor.execute("SELECT SUM(cost_usd), AVG(latency_sec), SUM(prompt_tokens + completion_tokens) FROM route_metrics")
    sum_cost, avg_latency, sum_tokens = cursor.fetchone()
    
    cursor.execute("SELECT COUNT(*) FROM route_metrics WHERE classified_complexity = 'simple'")
    simple_count = cursor.fetchone()[0]
    complex_count = total_queries - simple_count
    
    # Model distribution
    cursor.execute("SELECT chosen_model, COUNT(*) FROM route_metrics GROUP BY chosen_model")
    model_rows = cursor.fetchall()
    model_distribution = {row[0]: row[1] for row in model_rows}
    
    conn.close()
    
    return {
        "total_queries": total_queries,
        "total_cost": sum_cost or 0.0,
        "avg_latency": avg_latency or 0.0,
        "total_tokens": sum_tokens or 0,
        "simple_count": simple_count,
        "complex_count": complex_count,
        "model_distribution": model_distribution
    }

def get_detailed_transactions(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves detailed log history sorted by newest first."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM route_metrics ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Memory Manager

def save_chat_message(session_id: str, role: str, content: str):
    """Saves a single conversation history message."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, time.time())
    )
    conn.commit()
    conn.close()

def get_chat_messages(session_id: str) -> List[Dict[str, Any]]:
    """Retrieves all conversation history messages for a session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def trim_chat_messages(session_id: str, keep_last: int = 4):
    """Deletes oldest chat messages keeping only the recent ones."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Find the threshold timestamp of the message to keep
    cursor.execute(
        """
        SELECT timestamp FROM chat_messages 
        WHERE session_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 1 OFFSET ?
        """,
        (session_id, keep_last - 1)
    )
    row = cursor.fetchone()
    if row:
        threshold_time = row[0]
        cursor.execute(
            "DELETE FROM chat_messages WHERE session_id = ? AND timestamp <= ?",
            (session_id, threshold_time)
        )
        conn.commit()
    conn.close()

def get_running_summary(session_id: str) -> str:
    """Gets the running context summary paragraph for a session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM running_summaries WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return ""

def update_running_summary(session_id: str, summary: str):
    """Saves or updates a running summary context paragraph."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO running_summaries (session_id, summary, updated_at) 
        VALUES (?, ?, ?) 
        ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at
        """,
        (session_id, summary, time.time())
    )
    conn.commit()
    conn.close()
