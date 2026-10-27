# AML Memory API - Schema Temporal chain

JiuwenMemory PR #341 Schema TemporalEntity memory with the competition Add/Search HTTP API.

This repository is a self-contained source release for the Agent Memory Leaderboard Add/Search contract.
It contains the pinned JiuwenMemory source in `memory/` and the HTTP adapter in `adapter/`.

## Profile

- Schema extraction and entity association: enabled
- Every Search forces TemporalEntity dual recall
- Default temporal mode: history; deterministic time windows use range
- Source-first fallback: enabled

- Memory source revision: `98ac5171dbee29c7440d53afcf794ebaf39c134d`
- API: synchronous `POST /add` and `POST /search`; unauthenticated `GET /health`
- Authentication: `Bearer`, `Token`, or `X-Api-Key`
- Storage: Redis, Elasticsearch, and Milvus

## Start locally

Requirements: Docker with Compose and enough memory for Elasticsearch and Milvus. On Linux, set
`vm.max_map_count=262144` before starting Elasticsearch.

```powershell
Copy-Item .\server.env.example .\server.env
```

Edit `server.env` and set these values:

- `AML_API_KEY`: your own long random key for protecting Add/Search.
- `AML_LLM_BASE_URL`, `AML_LLM_API_KEY`, `AML_LLM_MODEL`.
- `AML_EMBED_BASE_URL`, `AML_EMBED_API_KEY`, `AML_EMBED_MODEL`.

Then build and start:

```powershell
docker compose --env-file .\server.env build api
docker compose --env-file .\server.env up -d
docker compose --env-file .\server.env ps
```

The container exposes the API only on `127.0.0.1:8000` by default.

## Verify Add/Search

```powershell
$env:AML_API_KEY = '<the value in server.env>'
python .\smoke.py --base-url http://127.0.0.1:8000
```

The smoke writes one synthetic memory, repeats the same `request_id` to check idempotency, searches
it, and verifies `user_id` isolation.

## Public HTTPS

Point a domain at the server, set `AML_PUBLIC_DOMAIN` in `server.env`, allow TCP 80/443, then run:

```powershell
docker compose --env-file .\server.env --profile public up -d
```

Caddy terminates HTTPS and proxies to the internal HTTP service. Do not expose Redis, Elasticsearch,
Milvus, etcd, or MinIO publicly.

## Add/Search contract

`POST /add` returns HTTP 200 only after extraction and configured indexes are written. A successful
response echoes `request_id`, `user_id`, and `session_id` and contains `success: true`.

`POST /search` accepts `query`, optional `options`, `user_id`, and `top_k`. It returns ranked
memory evidence in `data` and never generates the final benchmark answer.

See `openapi.json` for exact request and response schemas.

## Benchmark scripts

Five reproducible text-benchmark launchers are included for LongMemEval, PersonaMem-v2, CL-Bench,
BEAM, and LoCoMo-Refined. See [BENCHMARKS.md](BENCHMARKS.md) for setup, one-line commands, official
data sources, checkpoints, and output locations. The launchers are fixed to this repository's
`schema-temporal` profile, so every Search uses the forced temporal dual-recall path.

## Source and attribution

JiuwenMemory was originally developed by openJiuwen and is distributed under Apache-2.0. The upstream
source is <https://gitcode.com/openJiuwen/agent-memory>. Profile-specific provenance is recorded in
`SOURCE_PROVENANCE.md`. Adapter changes implement the competition HTTP boundary, persistence checks,
request idempotency, isolation, diagnostics, and fixed profile configuration.

Repository: <https://github.com/chin-alt/aml-memory-api-temporal>
