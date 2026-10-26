"""本机冒烟结构化诊断；不记录请求正文、响应正文、密钥或异常原文。"""

import json
import os
import re
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4


def endpoint_info(value):
    try:
        url = urlsplit(str(value))
        return {"scheme": url.scheme, "host": url.hostname, "port": url.port,
                "path": url.path, "query_omitted": bool(url.query)}
    except ValueError:
        return {"invalid_url": True}


def exception_info(exc):
    """保留完整因果类型链及代码位置，不格式化可能含正文/密钥的异常消息。"""
    chain, seen = [], set()
    current = exc
    while current is not None and id(current) not in seen and len(chain) < 12:
        seen.add(id(current))
        row = {"type": type(current).__name__, "module": type(current).__module__}
        for key in ("errno", "winerror", "status_code"):
            value = getattr(current, key, None)
            if isinstance(value, int):
                row[key] = value
        if (type(current).__name__ == "InvalidExtractionCandidateError"
                and type(current).__module__ == "jiuwen_memory.construction.extractor_impl.llm_extractor"):
            allowed = {"not_object", "missing_confidence", "invalid_confidence",
                       "confidence_out_of_range", "source_id_not_in_batch", "empty_content",
                       "non_string_evidence"}
            reasons = getattr(current, "reason_counts", None)
            if isinstance(reasons, dict):
                row["validation_reasons"] = {key: value for key, value in reasons.items()
                                             if key in allowed and isinstance(value, int)
                                             and not isinstance(value, bool) and value > 0}
        response = getattr(current, "response", None)
        if response is not None:
            row["status_code"] = response.status_code
        request = getattr(current, "request", None)
        if request is not None:
            row["endpoint"] = endpoint_info(request.url)
            row["method"] = request.method
        # 只接受简短机器错误码；不包含厂商 message / body。
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                row["provider_error"] = {k: v for k in ("code", "type")
                                         if isinstance(v := error.get(k), str)
                                         and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", v)}
        row["frames"] = [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                         for frame in traceback.extract_tb(current.__traceback__)[-16:]]
        chain.append(row)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    names = {row["type"] for row in chain}
    statuses = {row.get("status_code") for row in chain}
    if "ConnectTimeout" in names:
        hint = "Connection timed out: check gateway address/port, VPN, routing and proxy on the machine running this command."
    elif "ReadTimeout" in names:
        hint = "Connection established but response timed out: check gateway/upstream model latency."
    elif "ProxyError" in names:
        hint = "Proxy connection failed: inspect proxy environment and bypass rules."
    elif names & {"SSLError", "SSLCertVerificationError"}:
        hint = "TLS verification failed: inspect endpoint certificate and trusted CA."
    elif 401 in statuses or 403 in statuses:
        hint = "Authentication/authorization rejected: check the key and access to this gateway/model."
    elif 404 in statuses:
        hint = "Endpoint or model not found: check the API base path and model name."
    elif 429 in statuses:
        hint = "Upstream limit reached: check rate limits and quota."
    elif "InvalidExtractionCandidateError" in names:
        hint = "Extraction returned no usable candidates; inspect validation_reasons for field-level codes."
    else:
        hint = "Inspect stage, HTTP status, exception chain and code locations."
    return {"chain": chain, "hint": hint}


class DiagnosticLog:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()

    def emit(self, stage, event, **fields):
        if self.path is None:
            return
        row = {"time": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
               "thread": threading.get_ident(),
               "stage": stage, "event": event, **fields}
        text = json.dumps(row, ensure_ascii=True)
        # 防御性清理环境中的密钥；调用方仍必须只传诊断元数据。
        for key, value in os.environ.items():
            if value and any(word in key.upper() for word in ("API_KEY", "PASSWORD", "SECRET", "TOKEN")):
                text = text.replace(json.dumps(value)[1:-1], "[REDACTED]")
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(text + "\n")

    @contextmanager
    def span(self, stage, **fields):
        call_id = uuid4().hex[:12]
        started = time.perf_counter()
        self.emit(stage, "start", call_id=call_id, **fields)
        outcome = {}
        try:
            yield outcome
        except Exception as exc:
            self.emit(stage, "error", call_id=call_id,
                      elapsed_seconds=round(time.perf_counter() - started, 3),
                      error=exception_info(exc), **fields, **outcome)
            raise
        else:
            self.emit(stage, "ok", call_id=call_id,
                      elapsed_seconds=round(time.perf_counter() - started, 3), **fields, **outcome)


def wrap_model(instance, method, stage, diagnostic):
    """只包装当前已装配的实例；不改内核文件或全局 SDK 行为。"""
    if instance is None or getattr(instance, "_aml_diagnostic_wrapped", False):
        return
    original = getattr(instance, method)

    @wraps(original)
    def wrapped(*args, **kwargs):
        ep = instance._endpoint()
        from jiuwen_memory.common._support import OUTBOUND_CONNECT_TIMEOUT_SECONDS
        with diagnostic.span(stage, model=ep.model, endpoint=endpoint_info(ep.base_url),
                             timeout_seconds=instance._timeout,
                             connect_timeout_seconds=OUTBOUND_CONNECT_TIMEOUT_SECONDS,
                             sdk_max_retries=instance._max_retries,
                             input_count=len(args[0]) if args else None):
            return original(*args, **kwargs)

    setattr(instance, method, wrapped)
    instance._aml_diagnostic_wrapped = True


def environment_info(env):
    return {"proxy_env": {k: endpoint_info(v) for k, v in env.items()
                          if k.lower() in {"http_proxy", "https_proxy", "all_proxy"} and v},
            "no_proxy_present": bool(env.get("NO_PROXY") or env.get("no_proxy")),
            "note": "Environment snapshot only; proxy use is not inferred from this snapshot."}
