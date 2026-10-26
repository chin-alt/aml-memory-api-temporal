"""Schema experiment guards and feature-use diagnostics; no model or QA replacement."""

import os
from contextlib import contextmanager

from .diagnostics import DiagnosticLog


class SchemaEvaluation:
    def __init__(self, backend, ctx):
        from jiuwen_memory.construction.evolver import EvolverProducer
        from jiuwen_memory.construction.index_builder import IndexBuilderProducer
        from jiuwen_memory.storage.domain_store_impl.keyword_recaller import KeywordRecaller
        from jiuwen_memory.storage.entity_store import EntityStoreProducer

        self.log = DiagnosticLog(os.environ.get("AML_DIAGNOSTIC_LOG"))
        self.evolver = EvolverProducer.build_named("default", ctx)
        builder = IndexBuilderProducer.build_named("default", ctx)
        entity_builder = getattr(builder, "_entity_builder", None)
        if entity_builder is None:
            raise ValueError("Schema experiment requires an active entity index builder")
        entity_store = entity_builder._linker._entity_store
        storage = backend.api._engine._retriever.storage
        # Vector recall also uses L2; entity expansion belongs to keyword recall only.
        recallers = [r for r in storage.recallers if isinstance(r, KeywordRecaller) and r.layer == "l2"]
        if len(recallers) != 1 or recallers[0]._entity_store is not entity_store:
            raise ValueError("Entity recall and indexing must share the same entity store")
        # Explicitly create/validate the experiment index before ingest, not silently degrade.
        raw_store = EntityStoreProducer.build_named("default", ctx)
        raw_store.ensure_index()
        self.lookup_failed = False
        lookup = entity_store.find_by_entity_text_hash

        def checked_lookup(*args, **kwargs):
            try:
                with self.log.span("entity.lookup"):
                    return lookup(*args, **kwargs)
            except Exception:
                self.lookup_failed = True
                raise

        entity_store.find_by_entity_text_hash = checked_lookup
        recaller = recallers[0]
        expand = recaller._expand_by_entities

        def measured_expansion(*args, **kwargs):
            result = expand(*args, **kwargs)
            self.log.emit("entity.expand", "ok", added_candidates=len(result))
            return result

        recaller._expand_by_entities = measured_expansion
        self.log.emit("schema.profile", "ready", schema_enabled=True, entity_enabled=True)

    @contextmanager
    def writing(self):
        from jiuwen_memory.construction.source_update import STRICT_ENTITY_WRITES

        self.evolver.last_schema_error = ""
        self.last_write_degraded = False
        token = STRICT_ENTITY_WRITES.set(True)
        try:
            yield
            if self.evolver.last_schema_error:
                # The core deliberately preserves source evidence when one
                # schema extraction cannot be validated.  Keep that source-
                # first behavior and make the ablation visible in diagnostics
                # instead of converting a recoverable extraction miss into an
                # HTTP 503 that discards the rest of the conversation.
                self.last_write_degraded = True
                self.log.emit("schema.extraction", "degraded", source_only_fallback=True,
                              schema_error_present=True)
        finally:
            STRICT_ENTITY_WRITES.reset(token)

    def record_add(self, units):
        self.log.emit("schema.add", "ok", returned_units=len(units),
                      property_units=sum(bool(u.system_metadata.get("schema_property_name")) for u in units),
                      units_with_entities=sum(bool(u.entities) for u in units),
                      source_only_fallback=self.last_write_degraded)

    def begin_search(self):
        self.lookup_failed = False

    def end_search(self):
        if self.lookup_failed:
            raise RuntimeError("Entity lookup failed; refusing silent retrieval degradation")
