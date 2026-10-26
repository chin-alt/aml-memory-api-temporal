"""通过 mem2.0 公共 API 接入；生产配置限制为已检查写后可见性的后端。"""

import hashlib
import json
import os
from time import perf_counter
from datetime import datetime, timezone
from contextlib import nullcontext


class Mem2Backend:
    schema_eval = None

    def __init__(self, config_path, namespace, fidelity=False, smoke=False):
        from jiuwen_memory.api import assemble_runtime
        from jiuwen_memory.config import Config

        config = Config.from_yaml(str(config_path))
        ctx = config.context()
        self.namespace = namespace
        self.fidelity = fidelity
        self.persistent = not smoke
        self.repo_eval = os.environ.get("AML_REPO_EVAL", "false").lower() == "true"
        # The schema-temporal deployment profile is a separate immutable image.
        # In that image every public Search must enter PR #341's TemporalEntity
        # dual-recall path; it must not depend on automatic time-intent detection.
        self.schema_temporal_force = (
            ctx.globals.get("schema_temporal_enabled") is True
            and ctx.globals.get("schema_temporal_force_enabled") is True
        )
        self._evaluation_adds = {}
        self._evaluation_searches = {}
        self._validate(ctx, smoke)
        # fingerprint 不包含密钥；配置/模型/命名空间改变需使用新实验数据目录。
        specs = {name: {key: {"target": spec.target, "params": self._public(spec.params)}
                        for key, spec in group.items()} for name, group in ctx.namespaces.items()}
        settings = {"namespace": namespace, "fidelity": fidelity, "smoke": smoke,
                    "config": specs, "globals": self._public(ctx.globals), "adapter": 1,
                    "repository_evaluation": self.repo_eval}
        self.fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        self.runtime = assemble_runtime(config=config)
        self.api = self.runtime.api
        self.schema_eval = None
        # PR #337 needs the adapter's strict entity-recall diagnostics shim.  PR #341
        # has a native Schema TemporalEntity pipeline and must use that implementation
        # directly; wrapping it with the older experiment shim changes its semantics.
        if (ctx.globals.get("schema_enabled") is True
                and ctx.globals.get("entity_enabled") is True
                and ctx.globals.get("schema_temporal_enabled") is not True):
            from .schema_eval import SchemaEvaluation
            try:
                self.schema_eval = SchemaEvaluation(self, ctx)
            except Exception:
                self.runtime.close(wait=True)
                raise
        if os.environ.get("AML_DIAGNOSTIC_LOG") and not smoke:
            from .diagnostics import DiagnosticLog, wrap_model
            from jiuwen_memory.common.llm.base import LlmProducer
            from jiuwen_memory.common.embedder.base import EmbedderProducer
            diagnostic = DiagnosticLog(os.environ["AML_DIAGNOSTIC_LOG"])
            wrap_model(LlmProducer.build_named("default", ctx), "chat", "llm.chat", diagnostic)
            wrap_model(EmbedderProducer.build_named("default", ctx), "embed", "embedding.embed", diagnostic)

    @classmethod
    def _public(cls, value):
        if isinstance(value, dict):
            return {k: cls._public(v) for k, v in value.items()
                    if not any(s in k.lower() for s in ("api_key", "password", "token", "secret"))}
        if isinstance(value, list):
            return [cls._public(v) for v in value]
        return value

    @staticmethod
    def _validate(ctx, smoke):
        def spec(namespace, name="default"):
            value = ctx.namespaces.get(namespace, {}).get(name)
            if value is None:
                raise ValueError(f"Explicit {namespace}.{name} configuration required")
            return value

        if smoke:
            return
        schema = ctx.globals.get("schema_enabled") is True
        entity = ctx.globals.get("entity_enabled") is True
        temporal = ctx.globals.get("schema_temporal_enabled") is True
        if schema != entity:
            raise ValueError("Use either ordinary or the complete schema+entity profile")
        if temporal and not schema:
            raise ValueError("Schema Temporal retrieval requires schema+entity")
        for flag in ("graph_enabled",):
            if ctx.globals.get(flag) is not False:
                raise ValueError(f"This ordinary-chain profile requires {flag}=false")
        if schema:
            if spec("entity_store").target != "elasticsearch" or not spec("entity_store").params.get("hosts"):
                raise ValueError("Schema+entity requires an explicit Elasticsearch entity store")
            if not spec("extractor").params.get("schema_path"):
                raise ValueError("Schema+entity requires schema_path")
        if temporal:
            if ctx.globals.get("schema_temporal_force_enabled") is not True:
                raise ValueError("Competition Schema Temporal profile must force every Search")
            if spec("retriever").params.get("schema_temporal_auto_enabled") is not True:
                raise ValueError("Competition Schema Temporal profile requires automatic time intent")
            for namespace in ("fulltext_store", "vector_store"):
                spec(namespace, "schema_entities")
        for namespace, target in (("extractor", "entity_schema" if schema else "llm"),
                                  ("evolver", "schema_orchestrating" if schema else "orchestrating"),
                                  ("constructor", "hybrid"), ("kv_store", "redis"),
                                  ("embedder", "openai")):
            if spec(namespace).target != target:
                raise ValueError(f"Production adapter currently requires {namespace}={target}")
        if spec("llm").target not in {"openai", "dashscope"}:
            raise ValueError("Production adapter requires llm=openai or dashscope")
        for namespace, target in (("fulltext_store", "elasticsearch"), ("vector_store", "milvus")):
            for item in ctx.namespaces.get(namespace, {}).values():
                if item.target != target:
                    raise ValueError(f"Unsupported production backend for {namespace}")
                if namespace == "fulltext_store" and str(item.params.get("refresh")) not in ("wait_for", "true"):
                    raise ValueError("Elasticsearch requires refresh='wait_for' or 'true'")
                if namespace == "vector_store" and item.params.get("consistency_level") != "Strong":
                    raise ValueError("Milvus requires explicit Strong consistency")
        for namespace in ("fulltext_store", "vector_store"):
            spec(namespace)
        if ctx.globals.get("layers_index_enabled") is not False:
            # 本次交付默认关闭分层；避免继承内核的 memory 层索引却宣称持久化。
            raise ValueError("This adapter profile requires layers_index_enabled=false")
        for key in ("llm_api_key", "llm_base_url", "llm_model", "embedder_api_key",
                    "embedder_base_url", "embedder_model"):
            if not ctx.globals.get(key):
                raise ValueError(f"Missing required model setting: {key}")

    def scope(self, user_id, generation):
        from jiuwen_memory.api import Scope

        # 不把外部 ID 当路径拼接；所有 session 进入同一 user 范围。
        tenant = hashlib.sha256((self.namespace + "\0" + user_id).encode()).hexdigest()
        return Scope(org="aml_" + tenant, user=generation)

    @staticmethod
    def _temporal_search_extension(query):
        """Build the explicit PR #341 query used by every temporal-image Search.

        History is the safe default for general memory QA: it preserves old and
        updated property values.  A deterministic date window narrows that view
        to range mode without asking another model or silently disabling the
        TemporalEntity pipeline.
        """
        from jiuwen_memory.retrieval.schema_temporal import SchemaTimeExtractor

        window = SchemaTimeExtractor().extract(query)
        if not window.found:
            return {"mode": "history", "include_undated": True}
        value = {
            "mode": "range",
            "event_precision": window.precision.value,
            "include_undated": False,
        }
        if window.start is not None:
            value["event_from"] = window.start.isoformat()
        if window.end is not None:
            value["event_to"] = window.end.isoformat()
        return value

    def _search_context(self, scope, query):
        from jiuwen_memory.api import Context

        if not self.schema_temporal_force:
            return Context(scope)
        return Context(
            scope,
            extensions={"schema_temporal": self._temporal_search_extension(query)},
        )

    def add(self, request, generation):
        from jiuwen_memory.api import legacy_request_context

        scope = self.scope(request.user_id, generation)
        unit_ids = []
        for index, message in enumerate(request.messages):
            occurred = (datetime.fromtimestamp(message.timestamp / 1000, tz=timezone.utc)
                        if message.timestamp is not None else None)
            metadata = {
                "infer": True,
                "extraction_fidelity_mode": self.fidelity,
                "session_id": request.session_id,
                "request_id": request.request_id,
                "message_index": index,
                "role": message.role,
                # 不给缺少日期的评测历史凭空使用服务器今天作为观测日期。
                "observation_date": occurred.date().isoformat() if occurred else "unknown",
            }
            seed = getattr(request, "evaluation_seed", None)
            if seed is not None:
                if not getattr(self, "repo_eval", False):
                    raise ValueError("Evaluation metadata is not enabled")
                # 仓库数据使用无时区日期；不能让宿主机时区改变原始 QA 时间。
                occurred = datetime.fromisoformat(seed.occurred_at) if seed.occurred_at else None
                metadata = {**seed.metadata, "infer": True, "retain_source": True,
                            "extraction_fidelity_mode": self.fidelity,
                            "session_id": request.session_id, "request_id": request.request_id}
            with self.schema_eval.writing() if self.schema_eval else nullcontext():
                units = self.api.add(message.content if seed else f"[{message.role}]: {message.content}", scope,
                                     security=legacy_request_context(scope), tags=seed.tags if seed else None,
                                     system_metadata=metadata, occurred_at=occurred)
            if self.schema_eval:
                self.schema_eval.record_add(units)
            if getattr(self, "repo_eval", False):
                unit_ids.extend(unit.id for unit in units)
        if getattr(self, "repo_eval", False):
            self._evaluation_adds[(request.user_id, request.request_id)] = {"unit_ids": unit_ids}
        # 同步 add 已完成索引写入；ES wait_for + Milvus Strong 是发布成功的前提。

    def search(self, request, generation):
        from jiuwen_memory.api import DisclosureLevel, legacy_request_context

        scope = self.scope(request.user_id, generation)
        if self.schema_eval:
            self.schema_eval.begin_search()
        if getattr(self, "repo_eval", False):
            result = self._evaluation_search(request, scope)
            if self.schema_eval:
                self.schema_eval.end_search()
            return result
        # options 接受但基线不加入 query，不生成答案、不按选项挑记忆。
        result = self.api.search(request.query, self._search_context(scope, request.query),
                                 security=legacy_request_context(scope),
                                 top_k=request.top_k, disclosure=DisclosureLevel.L2)
        if result.errors:
            raise RuntimeError("Memory recall channel failed")
        if self.schema_eval:
            self.schema_eval.end_search()
        return [{"id": item.unit_id, "content": item.content, "score": item.score}
                for item in result.items]

    def evaluation_info(self, operation, request):
        if operation == "add":
            # 本地评测进程内重试可复用；进程重启后不能伪造写入映射。
            return self._evaluation_adds[(request.user_id, request.request_id)]
        return self._evaluation_searches.get(request.user_id, {
            "trajectory": [], "storage_recall_wall_ms": 0, "memory_retrieval_e2e_wall_ms": 0})

    def _evaluation_search(self, request, scope):
        from jiuwen_memory.api import DisclosureLevel, legacy_request_context
        from evaluation.longmemeval.artifacts import to_jsonable

        storage = self.api._engine._retriever.storage
        recall = storage.recall
        durations = []

        def timed(*args, **kwargs):
            started = perf_counter()
            try:
                return recall(*args, **kwargs)
            finally:
                durations.append((perf_counter() - started) * 1000)

        storage.recall = timed
        started = perf_counter()
        try:
            result = self.api.search(request.query, self._search_context(scope, request.query),
                                     security=legacy_request_context(scope),
                                     top_k=request.top_k, disclosure=DisclosureLevel.L2, with_trajectory=True)
        finally:
            elapsed = (perf_counter() - started) * 1000
            storage.recall = recall
        if result.errors or len(durations) != 1:
            raise RuntimeError("Repository evaluation recall failed or timing boundary changed")
        ids = [item.unit_id for item in result.items]
        units = self.api.inspect(ids, scope, security=legacy_request_context(scope)) if ids else []
        dates = {}
        for unit in units:
            temporal = unit.temporal
            dates[unit.id] = (temporal.t_message.isoformat() if temporal.t_message else "",
                              temporal.t_event.isoformat() if temporal.t_event else "")
        self._evaluation_searches[request.user_id] = {
            "trajectory": to_jsonable(result.trajectory), "memory_retrieval_e2e_wall_ms": elapsed,
            "storage_recall_wall_ms": durations[0]}
        return [{"id": item.unit_id, "content": item.content, "score": item.score,
                 "message_at": dates.get(item.unit_id, ("", ""))[0],
                 "event_at": dates.get(item.unit_id, ("", ""))[1],
                 "created_at": dates.get(item.unit_id, ("", ""))[0] or dates.get(item.unit_id, ("", ""))[1]}
                for item in result.items]

    def close(self):
        self.runtime.close(wait=True)
