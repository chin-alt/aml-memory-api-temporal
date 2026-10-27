"""Explicit memory profiles; evaluation prompts and sampled data stay shared."""

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("ordinary", "schema-entity", "schema-temporal",
            "bm25-fidelity", "score-max-fidelity")


def repository_root(profile):
    if profile not in PROFILES:
        raise ValueError("Unknown memory profile")
    standalone = ROOT.parent / "memory"
    if standalone.is_dir():
        return standalone
    folders = {
        "ordinary": "agent-memory",
        "schema-entity": "agent-memory-schema-pr337",
        "schema-temporal": "agent-memory-temporal-pr341",
        "bm25-fidelity": "agent-memory",
        "score-max-fidelity": "agent-memory",
    }
    path = ROOT.parent / folders[profile]
    if not path.is_dir():
        raise ValueError(f"Required repository checkout is missing: {path}")
    return path


def select_repository(profile):
    path = repository_root(profile)
    sys.path.insert(0, str(path))
    loaded = sys.modules.get("jiuwen_memory")
    if loaded and getattr(loaded, "__file__", None):
        if not Path(loaded.__file__).resolve().is_relative_to(path.resolve()):
            raise RuntimeError("Another memory checkout is already loaded; run this profile in a fresh process")
    return path


def profile_files(profile):
    if profile not in PROFILES:
        raise ValueError("Unknown memory profile")
    paths = [ROOT / "config/memory.yml"]
    if profile in ("schema-entity", "schema-temporal"):
        paths += [ROOT / "config/memory-schema-entity.yml",
                  repository_root(profile) / "examples/entity_schema_locomo_person.json"]
        if profile == "schema-temporal":
            paths.insert(2, ROOT / "config/memory-schema-temporal.yml")
    elif profile == "bm25-fidelity":
        paths.append(ROOT / "config/memory-bm25-fidelity.yml")
    elif profile == "score-max-fidelity":
        paths.append(ROOT / "config/memory-score-max-fidelity.yml")
    return paths


def memory_config(profile):
    paths = profile_files(profile)
    config = yaml.safe_load(paths[0].read_text(encoding="utf-8"))

    def merge(base, overlay):
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value

    overlays = []
    if profile == "schema-entity":
        overlays = paths[1:2]
    elif profile == "schema-temporal":
        overlays = paths[1:3]
    elif profile != "ordinary":
        overlays = paths[1:2]
    for overlay_path in overlays:
        merge(config, yaml.safe_load(overlay_path.read_text(encoding="utf-8")))
    if profile in ("schema-entity", "schema-temporal"):
        config["extractor"]["default"]["params"]["schema_path"] = str(paths[-1].resolve())
    return config
