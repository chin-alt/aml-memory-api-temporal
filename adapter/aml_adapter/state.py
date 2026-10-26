"""持久请求日志和 scope 代际切换；不假定多后端有分布式事务。"""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path


class ServiceError(Exception):
    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status = status
        self.code = code


class ProcessLock:
    """同一本地状态目录只允许一个进程，避免误开多 worker。"""

    def __init__(self, path: Path):
        self.file = path.open("a+b")
        self.file.seek(0, os.SEEK_END)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("State directory already in use; run exactly one worker") from None

    def close(self):
        self.file.close()


class Ledger:
    def __init__(self, path: Path, fingerprint: str):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY, generation TEXT NOT NULL, dirty INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL, request_id TEXT NOT NULL,
                digest TEXT NOT NULL, payload TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0,
                UNIQUE(user_id, request_id));
            CREATE INDEX IF NOT EXISTS requests_user ON requests(user_id, seq);
        """)
        row = self.db.execute("SELECT value FROM settings WHERE key='fingerprint'").fetchone()
        if row and row[0] != fingerprint:
            self.db.close()
            raise RuntimeError("Backend/config changed; use a new state directory and namespace")
        self.db.execute("INSERT OR IGNORE INTO settings VALUES ('fingerprint', ?)", (fingerprint,))
        self.db.commit()

    @contextmanager
    def transaction(self):
        with self.lock, self.db:
            yield self.db

    def prepare(self, request, digest: str, generation: str):
        """返回（旧代际，之前是否脏，是否已成功）；先持久化意图，再允许写后端。"""
        with self.transaction() as db:
            old = db.execute(
                "SELECT * FROM requests WHERE user_id=? AND request_id=?",
                (request.user_id, request.request_id),
            ).fetchone()
            if old and old["digest"] != digest:
                raise ServiceError(409, "request_id_payload_conflict")
            db.execute("INSERT OR IGNORE INTO users VALUES (?, ?, 0)",
                       (request.user_id, generation))
            user = db.execute("SELECT * FROM users WHERE user_id=?", (request.user_id,)).fetchone()
            if old and old["done"] and not user["dirty"]:
                return user["generation"], False, True
            if old is None:
                db.execute(
                    "INSERT INTO requests(user_id, request_id, digest, payload) VALUES (?,?,?,?)",
                    (request.user_id, request.request_id, digest, request.model_dump_json()),
                )
            db.execute("UPDATE users SET dirty=1 WHERE user_id=?", (request.user_id,))
            return user["generation"], bool(user["dirty"]), False

    def user(self, user_id):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
            return dict(row) if row else None

    def history(self, user_id):
        with self.transaction() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT payload FROM requests WHERE user_id=? ORDER BY seq", (user_id,))]

    def finish(self, user_id, generation):
        with self.transaction() as db:
            db.execute("UPDATE requests SET done=1 WHERE user_id=?", (user_id,))
            db.execute("UPDATE users SET generation=?, dirty=0 WHERE user_id=?",
                       (generation, user_id))

    def invalidate_all(self):
        with self.transaction() as db:
            db.execute("UPDATE users SET dirty=1")

    def close(self):
        self.db.close()
