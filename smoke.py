"""Smoke a deployed competition Add/Search endpoint using synthetic text."""

import argparse
import json
import os
import urllib.request
from uuid import uuid4


def request(base, key, path, payload=None, timeout=1800):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    key = os.environ["AML_API_KEY"]
    user = "deployment-smoke-" + uuid4().hex
    status, health = request(args.base_url, key, "/health")
    assert status == 200 and health["status"] == "ok"
    payload = {
        "request_id": user + ":1",
        "user_id": user,
        "session_id": "session-1",
        "messages": [{"role": "user", "content": "My bicycle is purple."}],
    }
    status, added = request(args.base_url, key, "/add", payload)
    assert status == 200 and added["success"] is True
    _, retry = request(args.base_url, key, "/add", payload)
    assert retry == added
    _, found = request(
        args.base_url, key, "/search",
        {"query": "What color is my bicycle?", "user_id": user, "top_k": 10},
    )
    assert found["data"] and any("purple" in row["content"].lower() for row in found["data"])
    _, isolated = request(
        args.base_url, key, "/search",
        {"query": "bicycle", "user_id": user + "-other", "top_k": 10},
    )
    assert isolated == {"data": []}
    print("PASS: health, Add, idempotent retry, Search, user isolation")


if __name__ == "__main__":
    main()
