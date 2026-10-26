"""顺序写入、幂等重试及中断恢复。恢复时重建新 scope，成功后原子发布。"""

import hashlib
import json
import math
import threading
from contextlib import contextmanager
from uuid import uuid4

from .models import AddRequest
from .state import ServiceError


class MemoryService:
    def __init__(self, backend, ledger, max_concurrency=4, request_type=AddRequest):
        self.backend = backend
        self.request_type = request_type
        self.ledger = ledger
        self.slots = threading.BoundedSemaphore(max_concurrency)
        self.stripes = [threading.Lock() for _ in range(257)]
        if not backend.persistent:
            ledger.invalidate_all()

    @contextmanager
    def admission(self, user_id, operation):
        index = int.from_bytes(hashlib.sha256(user_id.encode()).digest()[:4], "big")
        lock = self.stripes[index % len(self.stripes)]
        if not self.slots.acquire(blocking=False):
            raise ServiceError(429, "capacity_exceeded")
        acquired = lock.acquire(blocking=False)
        try:
            if not acquired:
                raise ServiceError(409 if operation == "add" else 425, "user_busy")
            yield
        finally:
            if acquired:
                lock.release()
            self.slots.release()

    @staticmethod
    def response(request):
        return {"success": True, "request_id": request.request_id,
                "user_id": request.user_id, "session_id": request.session_id}

    def rebuild(self, user_id):
        # 任何失败/崩溃都不能在可能已部分写入的旧 scope 上盲目重放。
        generation = uuid4().hex
        for payload in self.ledger.history(user_id):
            self.backend.add(self.request_type.model_validate(payload), generation)
        self.ledger.finish(user_id, generation)
        return generation

    def add(self, request):
        with self.admission(request.user_id, "add"):
            canonical = json.dumps(request.model_dump(), sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"))
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            generation, dirty, done = self.ledger.prepare(request, digest, uuid4().hex)
            if done:
                return self.response(request)
            if dirty:
                self.rebuild(request.user_id)
            else:
                self.backend.add(request, generation)
                self.ledger.finish(request.user_id, generation)
            return self.response(request)

    def search(self, request):
        with self.admission(request.user_id, "search"):
            user = self.ledger.user(request.user_id)
            if user is None:
                return {"data": []}
            generation = self.rebuild(request.user_id) if user["dirty"] else user["generation"]
            items = self.backend.search(request, generation)
            data, seen = [], set()
            for item in items:
                if not isinstance(item.get("id"), str) or not item["id"].strip():
                    raise RuntimeError("backend returned invalid memory id")
                if not isinstance(item.get("content"), str) or not item["content"].strip():
                    raise RuntimeError("backend returned invalid memory content")
                if item["id"] in seen:
                    continue
                seen.add(item["id"])
                record = {"id": item["id"], "content": item["content"]}
                score = item.get("score")
                if isinstance(score, (float, int)) and not isinstance(score, bool) and math.isfinite(score):
                    record["score"] = score
                if item.get("created_at"):
                    record["created_at"] = item["created_at"]
                for field in ("message_at", "event_at"):
                    if field in item:
                        record[field] = item[field]
                data.append(record)
                if len(data) == request.top_k:
                    break
            return {"data": data}
