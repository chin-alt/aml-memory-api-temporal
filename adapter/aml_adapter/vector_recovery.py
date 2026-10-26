"""Rebuild the lost Milvus L2 projection from persisted Elasticsearch documents."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .lme_profiles import select_repository


def _flatten_metadata(metadata: dict) -> dict:
    """Invert ElasticsearchFulltextStore._source for vector metadata."""
    flattened = {}
    for key, value in metadata.items():
        if key in {"system_metadata", "user_metadata"} and isinstance(value, dict):
            flattened.update({f"{key}.{nested}": item for nested, item in value.items()})
        else:
            flattened[key] = value
    return flattened


def _scope_pairs(database: Path) -> tuple[list[tuple[str, str]], list[str]]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT user_id, generation, dirty FROM users ORDER BY user_id"
        ).fetchall()
    finally:
        connection.close()
    clean = [(user_id, generation) for user_id, generation, dirty in rows if not dirty]
    dirty = [user_id for user_id, _, is_dirty in rows if is_dirty]
    return clean, dirty


def _fetch_documents(es_url: str, index: str, scopes: list[dict]) -> list[dict]:
    if not scopes:
        return []
    clauses = [
        {"bool": {"filter": [
            {"term": {"scope.org": scope["org"]}},
            {"term": {"scope.user": scope["user"]}},
        ]}}
        for scope in scopes
    ]
    body = {
        "size": 1000,
        "query": {"bool": {"should": clauses, "minimum_should_match": 1}},
        "_source": ["logical_id", "text", "scope", "metadata"],
        "sort": ["_doc"],
    }
    documents = []
    with httpx.Client(base_url=es_url.rstrip("/"), timeout=60, trust_env=False) as client:
        response = client.post(f"/{index}/_search", params={"scroll": "5m"}, json=body)
        response.raise_for_status()
        page = response.json()
        scroll_id = page.get("_scroll_id")
        try:
            while True:
                hits = page.get("hits", {}).get("hits", [])
                if not hits:
                    break
                documents.extend(hit["_source"] for hit in hits)
                response = client.post("/_search/scroll", json={
                    "scroll": "5m", "scroll_id": scroll_id,
                })
                response.raise_for_status()
                page = response.json()
                scroll_id = page.get("_scroll_id", scroll_id)
        finally:
            if scroll_id:
                try:
                    client.request("DELETE", "/_search/scroll", json={"scroll_id": [scroll_id]})
                except httpx.HTTPError:
                    pass
    return documents


def _validate_document(source: dict, allowed_scopes: set[tuple[str, str]]) -> None:
    logical_id = source.get("logical_id")
    text = source.get("text")
    scope = source.get("scope")
    metadata = source.get("metadata")
    if not isinstance(logical_id, str) or not logical_id:
        raise ValueError("Elasticsearch recovery document has no logical_id")
    if not isinstance(text, str) or not text:
        raise ValueError(f"Elasticsearch recovery document {logical_id} has no text")
    if not isinstance(scope, dict) or any(
        not isinstance(scope.get(key), str)
        for key in ("org", "space", "user", "agent", "session")
    ):
        raise ValueError(f"Elasticsearch recovery document {logical_id} has incomplete scope")
    if (scope["org"], scope["user"]) not in allowed_scopes:
        raise ValueError(f"Elasticsearch recovery document {logical_id} is outside active scope")
    required = {
        "unit_id", "tier", "lifecycle", "tags", "entities", "source",
        "content_layer", "t_event", "t_invalid",
    }
    if not isinstance(metadata, dict) or required - metadata.keys():
        missing = sorted(required - set(metadata or {}))
        raise ValueError(f"Elasticsearch recovery document {logical_id} lacks {missing}")
    if metadata["unit_id"] != logical_id or metadata["content_layer"] != "l2":
        raise ValueError(f"Elasticsearch recovery document {logical_id} has inconsistent identity/layer")


def recover_vector_store(
    output: Path,
    state: dict,
    profile: str,
    env: dict,
    *,
    progress_every: int = 100,
) -> dict:
    """Restore clean published generations without invoking the memory extractor.

    Dirty generations are deliberately excluded. Their current scope can contain a
    partially applied Add, so the normal ledger rebuild must publish a fresh generation.
    """
    database = output / "http-state" / "requests.sqlite3"
    if not database.is_file():
        raise ValueError("Vector recovery requires the persisted request ledger")
    clean_users, dirty_users = _scope_pairs(database)
    user_to_conversation = {
        record["user_id"]: ident for ident, record in state["conversations"].items()
    }
    unknown = [user_id for user_id, _ in clean_users if user_id not in user_to_conversation]
    if unknown:
        raise ValueError("Vector recovery ledger contains users outside this run")

    import hashlib

    namespace = state["namespace"]
    scopes = []
    for user_id, generation in clean_users:
        tenant = hashlib.sha256((namespace + "\0" + user_id).encode()).hexdigest()
        scopes.append({
            "org": "aml_" + tenant,
            "space": "",
            "user": generation,
            "agent": "",
            "session": "",
            "external_user_id": user_id,
        })

    index = env["AML_ES_INDEX"]
    documents = _fetch_documents(env["AML_ES_URL"], index, scopes)
    allowed = {(scope["org"], scope["user"]) for scope in scopes}
    for source in documents:
        _validate_document(source, allowed)
    identities = {
        (
            tuple(source["scope"][key] for key in ("org", "space", "user", "agent", "session")),
            source["logical_id"],
        )
        for source in documents
    }
    if len(identities) != len(documents):
        raise ValueError("Elasticsearch recovery contains duplicate scope/logical_id pairs")
    if clean_users and not documents:
        raise ValueError("Elasticsearch has no documents for the clean published generations")

    select_repository(profile)
    from jiuwen_memory.common.chunker.chunker_impl.fixed_window_chunker import FixedWindowChunker
    from jiuwen_memory.common.embedder.embedder_impl.openai_embedder import OpenAIEmbedder
    from jiuwen_memory.common.type_def import Scope
    from jiuwen_memory.storage.kv_impl.redis_kv import RedisKVStore
    from jiuwen_memory.storage.types import VectorRecord
    from jiuwen_memory.storage.vector_impl.milvus_vector import MilvusVectorStore

    dimension = int(env.get("AML_EMBED_DIM", "1024"))
    chunk_size = int(env.get("AML_CHUNK_SIZE", "1024"))
    max_batch = int(env.get("AML_EMBED_MAX_BATCH", "10"))
    embedder = OpenAIEmbedder(
        model_name=env["AML_EMBED_MODEL"],
        base_url=env["AML_EMBED_BASE_URL"],
        api_key=env["AML_EMBED_API_KEY"],
        dimension=dimension,
        max_batch_size=max_batch,
        timeout=float(env.get("AML_EMBED_TIMEOUT", "60")),
        max_retries=int(env.get("AML_EMBED_MAX_RETRIES", "1")),
        ssl_verify=urlsplit(env["AML_EMBED_BASE_URL"]).scheme == "https",
    )
    chunker = FixedWindowChunker(size=chunk_size)
    vector_store = MilvusVectorStore(
        uri=env["AML_MILVUS_URI"],
        collection=env["AML_MILVUS_COLLECTION"],
        dim=dimension,
        metric_type="COSINE",
        consistency_level="Strong",
    )
    kv_store = RedisKVStore(url=env["AML_REDIS_URL"])

    pending = []
    tracking_by_unit: dict[tuple[tuple[str, str, str, str, str], str], list[str]] = {}
    for source in documents:
        metadata = _flatten_metadata(source["metadata"])
        vector_text = source["text"]
        if metadata.get("tier") == "procedural":
            vector_text = metadata.get("system_metadata.searchable_text") or vector_text
        chunks = chunker.chunk(vector_text, source["logical_id"], {"tier": metadata["tier"]})
        if not chunks:
            raise ValueError(f"Chunker produced no chunks for {source['logical_id']}")
        scope = Scope(**source["scope"])
        scope_key = tuple(source["scope"][key]
                          for key in ("org", "space", "user", "agent", "session"))
        tracking_by_unit[(scope_key, source["logical_id"])] = [
            f"{source['logical_id']}-{chunk.id}" for chunk in chunks
        ]
        pending.extend((scope, source["logical_id"], metadata, chunk) for chunk in chunks)

    written = 0
    for offset in range(0, len(pending), max_batch):
        batch = pending[offset:offset + max_batch]
        last_error = None
        for attempt in range(1, 6):
            try:
                vectors = embedder.embed([item[3].text for item in batch])
                break
            except Exception as error:
                last_error = error
                if attempt == 5:
                    raise
                delay = min(30, attempt * 2)
                print(f"RETRY vector recovery embedding: attempt={attempt}/5; wait={delay}s; "
                      f"error={type(error).__name__}", flush=True)
                time.sleep(delay)
        else:  # pragma: no cover - loop either succeeds or raises
            raise last_error
        if len(vectors) != len(batch) or any(len(vector) != dimension for vector in vectors):
            raise ValueError("Embedding service returned an unexpected vector count or dimension")
        rows = []
        for (scope, unit_id, metadata, chunk), vector in zip(batch, vectors):
            record_id = f"{unit_id}-{chunk.id}"
            record = VectorRecord(
                id=record_id,
                vector=vector,
                metadata={**metadata, "seq": chunk.seq},
            )
            rows.append(vector_store._row(scope, record))
        vector_store.client.upsert(env["AML_MILVUS_COLLECTION"], data=rows)
        written += len(rows)
        if written == len(pending) or written % progress_every < len(batch):
            print(f"VECTOR RECOVERY: {written}/{len(pending)} rows", flush=True)

    pipeline = kv_store.client.pipeline(transaction=False)
    for (scope_key, unit_id), record_ids in tracking_by_unit.items():
        scope = Scope(org=scope_key[0], space=scope_key[1], user=scope_key[2],
                      agent=scope_key[3], session=scope_key[4])
        key = kv_store._namespaced(scope, f"/index/chunks/{unit_id}")
        pipeline.set(key, json.dumps(record_ids).encode())
    pipeline.execute()

    expected_by_scope: dict[tuple[str, str], int] = {}
    for scope, _, _, _ in pending:
        key = (scope.org, scope.user)
        expected_by_scope[key] = expected_by_scope.get(key, 0) + 1
    verified = 0
    client = vector_store.client
    for (org, generation), expected in expected_by_scope.items():
        rows = client.query(
            env["AML_MILVUS_COLLECTION"],
            filter=f'scope_org == "{org}" && scope_user == "{generation}"',
            output_fields=["logical_id"],
            limit=max(16384, expected),
            consistency_level="Strong",
        )
        ids = {row["logical_id"] for row in rows}
        if len(ids) != expected:
            raise RuntimeError(
                f"Milvus recovery verification failed for generation {generation}: "
                f"expected={expected}, actual={len(ids)}"
            )
        verified += len(ids)

    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "elasticsearch-direct-vector-rebuild",
        "elasticsearch_index": index,
        "milvus_collection": env["AML_MILVUS_COLLECTION"],
        "embedding_model": env["AML_EMBED_MODEL"],
        "dimension": dimension,
        "chunk_size": chunk_size,
        "clean_users": len(clean_users),
        "dirty_users_skipped": dirty_users,
        "documents": len(documents),
        "vector_rows": len(pending),
        "verified_rows": verified,
    }
