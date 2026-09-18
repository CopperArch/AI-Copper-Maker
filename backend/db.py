import sqlite3
import json
import threading
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "trainer.db"

_schema = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    summary_message_id TEXT DEFAULT '',
    cost REAL NOT NULL DEFAULT 0.0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_calls TEXT DEFAULT '[]',
    finish_reason TEXT DEFAULT '',
    created_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);

CREATE TABLE IF NOT EXISTS usage (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost REAL NOT NULL DEFAULT 0.0,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS permissions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_args TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at INTEGER NOT NULL,
    responded_at INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
"""

def init_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_schema)
    conn.commit()
    return conn

def _migrate_from_json(conn: sqlite3.Connection):
    json_path = Path(__file__).parent.parent / "conversations.json"
    if not json_path.exists():
        return 0
    try:
        data = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return 0
    if not isinstance(data, list):
        return 0
    count = 0
    for session in data:
        sid = session.get("id")
        if not sid:
            continue
        title = session.get("title", "")
        model = session.get("model", "")
        created = session.get("created_at", 0)
        updated = session.get("updated_at", 0)
        summary_msg_id = session.get("summary_message_id", "")
        cost = session.get("cost", 0.0)
        prompt_tokens = session.get("prompt_tokens", 0)
        completion_tokens = session.get("completion_tokens", 0)
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, title, model, created_at, updated_at, summary_message_id, cost, prompt_tokens, completion_tokens) VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, title, model, created, updated, summary_msg_id, cost, prompt_tokens, completion_tokens)
        )
        for msg in session.get("messages", []):
            mid = msg.get("id") or f"{sid}_{msg.get('role', 'user')}_{msg.get('created_at', 0)}"
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = json.dumps(msg.get("tool_calls", []))
            finish = msg.get("finish_reason", "")
            created_at = msg.get("created_at", 0)
            conn.execute(
                "INSERT OR REPLACE INTO messages (id, session_id, role, content, tool_calls, finish_reason, created_at) VALUES (?,?,?,?,?,?,?)",
                (mid, sid, role, content, tool_calls, finish, created_at)
            )
        count += 1
    conn.commit()
    return count

def get_conn() -> sqlite3.Connection:
    return init_db()

def create_session(conn: sqlite3.Connection, sid: str, title: str, model: str = "") -> str:
    now = int(datetime.now().timestamp())
    conn.execute(
        "INSERT OR REPLACE INTO sessions (id, title, model, created_at, updated_at, summary_message_id, cost, prompt_tokens, completion_tokens) VALUES (?,?,?,?,?,?,?,?,?)",
        (sid, title, model, now, now, "", 0.0, 0, 0)
    )
    conn.commit()
    return sid

def get_session(conn: sqlite3.Connection, sid: str) -> dict | None:
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
    if row is None:
        return None
    return dict(row)

def get_messages(conn: sqlite3.Connection, sid: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC", (sid,)
    ).fetchall()
    return [dict(r) for r in rows]

def save_message(conn: sqlite3.Connection, mid: str, sid: str, role: str, content: str = "", tool_calls: list = None, finish_reason: str = "", created_at: int = None):
    if created_at is None:
        created_at = int(datetime.now().timestamp())
    if tool_calls is None:
        tool_calls = []
    if role in ("tool_call", "tool_result") and not isinstance(content, str):
        content = json.dumps(content)
    conn.execute(
        "INSERT OR REPLACE INTO messages (id, session_id, role, content, tool_calls, finish_reason, created_at) VALUES (?,?,?,?,?,?,?)",
        (mid, sid, role, content, json.dumps(tool_calls), finish_reason, created_at)
    )
    conn.commit()

def update_session_summary(conn: sqlite3.Connection, sid: str, summary_message_id: str):
    conn.execute("UPDATE sessions SET summary_message_id = ?, updated_at = ? WHERE id = ?", (summary_message_id, int(datetime.now().timestamp()), sid))
    conn.commit()

def update_session_usage(conn: sqlite3.Connection, sid: str, input_tokens: int, output_tokens: int, cost: float):
    now = int(datetime.now().timestamp())
    conn.execute("UPDATE sessions SET prompt_tokens = prompt_tokens + ?, completion_tokens = completion_tokens + ?, cost = cost + ?, updated_at = ? WHERE id = ?", (input_tokens, output_tokens, cost, now, sid))
    conn.commit()

def delete_session(conn: sqlite3.Connection, sid: str):
    conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
    conn.execute("DELETE FROM usage WHERE session_id = ?", (sid,))
    conn.execute("DELETE FROM permissions WHERE session_id = ?", (sid,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    conn.commit()

def create_permission(conn: sqlite3.Connection, pid: str, sid: str, tool_name: str, tool_args: dict) -> str:
    now = int(datetime.now().timestamp())
    conn.execute("INSERT INTO permissions (id, session_id, tool_name, tool_args, status, created_at) VALUES (?,?,?,?,?,?)", (pid, sid, tool_name, json.dumps(tool_args), "pending", now))
    conn.commit()
    return pid

def get_permission(conn: sqlite3.Connection, pid: str) -> dict | None:
    row = conn.execute("SELECT * FROM permissions WHERE id = ?", (pid,)).fetchone()
    if row is None:
        return None
    return dict(row)

def respond_permission(conn: sqlite3.Connection, pid: str, approved: bool):
    now = int(datetime.now().timestamp())
    conn.execute("UPDATE permissions SET status = ?, responded_at = ? WHERE id = ?", ("approved" if approved else "denied", now, pid))
    conn.commit()

def get_pending_permissions(conn: sqlite3.Connection, sid: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM permissions WHERE session_id = ? AND status = 'pending'", (sid,)).fetchall()
    return [dict(r) for r in rows]

def _msg_to_frontend(m: dict) -> dict:
    role = m.get("role", "user")
    content = m.get("content", "")
    if role in ("tool_call", "tool_result"):
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            payload = {}
        if role == "tool_call":
            return {"role": "tool_call",
                    "name": payload.get("name", ""),
                    "arguments": payload.get("arguments", {})}
        return {"role": "tool_result",
                "name": payload.get("name", ""),
                "result": payload.get("result", "")}
    return {"role": role, "content": content}

def sessions_list(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
    sessions = []
    for r in rows:
        s = dict(r)
        msgs = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC, rowid ASC",
            (s["id"],),
        ).fetchall()
        s["messages"] = [_msg_to_frontend(dict(m)) for m in msgs]
        sessions.append(s)
    return sessions
