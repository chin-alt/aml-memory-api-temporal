"""部署入口：uvicorn aml_adapter.app:create_app --factory --workers 1。"""

import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .backend import Mem2Backend
from .models import AddRequest, AddResponse, SearchRequest, SearchResponse
from .models import EvaluationAddRequest, EvaluationSearchRequest
from .service import MemoryService
from .state import Ledger, ProcessLock, ServiceError
from .diagnostics import DiagnosticLog

logger = logging.getLogger("aml_adapter")


class BodyLimit:
    """有界接收，不静默截断。纯 ASGI，避免长任务被 HTTP middleware 取消。"""
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                return await JSONResponse({"error": "payload_too_large"}, 413)(scope, receive, send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        sent = False

        async def buffered():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, buffered, send)


def create_app(*, backend=None, state_dir=None, api_key=None, concurrency=None):
    key = api_key if api_key is not None else os.environ.get("AML_API_KEY", "")
    if not key:
        raise RuntimeError("Set AML_API_KEY (participant Memory System Key, not Eval Key)")
    directory = Path(state_dir or os.environ.get("AML_STATE_DIR", "./state"))
    capacity = concurrency or int(os.environ.get("AML_MAX_CONCURRENCY", "4"))
    diagnostic = DiagnosticLog(os.environ.get("AML_DIAGNOSTIC_LOG"))
    repo_eval = os.environ.get("AML_REPO_EVAL", "false").lower() == "true"
    if repo_eval and capacity != 1:
        raise ValueError("Repository evaluation HTTP mode requires AML_MAX_CONCURRENCY=1")
    if capacity < 1:
        raise ValueError("AML_MAX_CONCURRENCY must be positive")

    @asynccontextmanager
    async def lifespan(app):
        directory.mkdir(parents=True, exist_ok=True)
        process_lock = ProcessLock(directory / "service.lock")
        active, ledger = backend, None
        try:
            if active is None:
                active = Mem2Backend(
                    os.environ.get("AML_MEMORY_CONFIG", "config/memory.yml"),
                    os.environ.get("AML_NAMESPACE", "aml-text-v1"),
                    fidelity=os.environ.get("AML_EXTRACTION_FIDELITY", "false").lower() == "true",
                    smoke=os.environ.get("AML_LOCAL_SMOKE", "false").lower() == "true",
                )
            ledger = Ledger(directory / "requests.sqlite3", active.fingerprint)
            app.state.service = MemoryService(active, ledger, capacity,
                                              request_type=EvaluationAddRequest if repo_eval else AddRequest)
            yield
        finally:
            if active is not None:
                active.close()
            if ledger is not None:
                ledger.close()
            process_lock.close()

    app = FastAPI(title="AML mem2.0 Text API", version="1.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None)
    app.add_middleware(BodyLimit, limit=int(os.environ.get("AML_MAX_BODY_BYTES", "16777216")))

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path == "/health" and request.method == "GET":
            return await call_next(request)
        authorization = request.headers.get("authorization", "").split(" ", 1)
        supplied = request.headers.get("x-api-key", "")
        if len(authorization) == 2 and authorization[0].lower() in {"bearer", "token"}:
            supplied = authorization[1]
        if not hmac.compare_digest(supplied.encode(), key.encode()):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Pydantic 默认错误含 input，可能把整段评测文本反射到响应/日志。
        return JSONResponse({"error": "invalid_request", "fields": [
            {"loc": list(error["loc"]), "type": error["type"]} for error in exc.errors()
        ]}, status_code=422)

    def execute(operation, payload):
        try:
            with diagnostic.span("api." + operation):
                result = getattr(app.state.service, operation)(payload)
                if repo_eval:
                    result["evaluation"] = app.state.service.backend.evaluation_info(operation, payload)
                return result
        except ServiceError as exc:
            return JSONResponse({"error": exc.code}, status_code=exc.status,
                                headers={"Retry-After": "2"})
        except Exception as exc:
            logger.warning("%s failed (%s); payload omitted", operation, type(exc).__name__)
            return JSONResponse({"error": "backend_unavailable"}, status_code=503,
                                headers={"Retry-After": "2"})

    @app.get("/health")
    def health():
        # 存活检查；真实后端/模型连接由本地或平台 smoke 验证。
        return {"status": "ok", "version": "1.0.0"}

    if repo_eval:
        @app.post("/add")
        def evaluation_add(payload: EvaluationAddRequest):
            return execute("add", payload)

        @app.post("/search")
        def evaluation_search(payload: EvaluationSearchRequest):
            return execute("search", payload)
    else:
        @app.post("/add", response_model=AddResponse)
        def add(payload: AddRequest):
            return execute("add", payload)

        @app.post("/search", response_model=SearchResponse, response_model_exclude_none=True)
        def search(payload: SearchRequest):
            return execute("search", payload)

    return app
