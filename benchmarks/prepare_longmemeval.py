"""Download the pinned public LongMemEval Oracle split and freeze the 30-case sample."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from aml_adapter.lme_sampling import prepare

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "adapter/data/longmemeval_oracle.json"
OUTPUT = ROOT / "adapter/data/lme-oracle-30-answerable-seed42"
URL = ("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
       "longmemeval_oracle.json")
SHA256 = "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        temporary = DATA.with_suffix(".json.download")
        request = urllib.request.Request(URL, headers={"User-Agent": "AML-benchmark-runner"})
        with urllib.request.urlopen(request, timeout=600) as response:
            temporary.write_bytes(response.read())
        temporary.replace(DATA)
    if digest(DATA) != SHA256:
        raise ValueError("LongMemEval Oracle checksum mismatch")
    if not (OUTPUT / "manifest.json").exists():
        prepare(DATA, OUTPUT, per_type=5, seed=42, variant="oracle", answerable_only=True)
    print("LongMemEval manifest: " + str(OUTPUT / "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
