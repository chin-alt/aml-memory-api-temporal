"""Render the immutable memory profile baked into a deployment image."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("ordinary", "schema-entity", "schema-temporal")


def merge(base: dict, overlay: dict) -> dict:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def render(profile: str, schema_path: str = "/app/agent-memory/examples/entity_schema_locomo_person.json") -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Unsupported deployment profile: {profile}")
    config = yaml.safe_load((ROOT / "config/memory.yml").read_text(encoding="utf-8"))
    if profile in {"schema-entity", "schema-temporal"}:
        overlay = yaml.safe_load(
            (ROOT / "config/memory-schema-entity.yml").read_text(encoding="utf-8")
        )
        config = merge(config, overlay)
        if profile == "schema-temporal":
            temporal = yaml.safe_load(
                (ROOT / "config/memory-schema-temporal.yml").read_text(encoding="utf-8")
            )
            config = merge(config, temporal)
        config["extractor"]["default"]["params"]["schema_path"] = schema_path
        config["entity_store"]["default"]["params"]["index"] = (
            "${AML_ENTITY_ES_INDEX:-aml_schema_entities_v1}"
        )
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schema-path",
        default="/app/agent-memory/examples/entity_schema_locomo_person.json",
    )
    args = parser.parse_args(argv)
    if args.profile in {"schema-entity", "schema-temporal"} and not Path(args.schema_path).is_file():
        raise ValueError(f"Schema file is missing: {args.schema_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(render(args.profile, args.schema_path), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
