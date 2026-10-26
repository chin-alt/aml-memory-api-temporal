"""Explicit memory profiles; evaluation prompts and sampled data stay shared."""

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("ordinary", "schema-entity", "bm25-fidelity", "score-max-fidelity")


def repository_root(profile):
    if profile not in PROFILES:
        raise ValueError("Unknown memory profile")
    folder = "agent-memory-schema-pr337" if profile == "schema-entity" else "agent-memory"
    path = ROOT.parent / folder
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
    if profile == "schema-entity":
        paths += [ROOT / "config/memory-schema-entity.yml",
                  repository_root(profile) / "examples/entity_schema_locomo_person.json"]
    elif profile == "bm25-fidelity":
        paths.append(ROOT / "config/memory-bm25-fidelity.yml")
    elif profile == "score-max-fidelity":
        paths.append(ROOT / "config/memory-score-max-fidelity.yml")
    return paths


def memory_config(profile):
    paths = profile_files(profile)
    config = yaml.safe_load(paths[0].read_text(encoding="utf-8"))
    if profile != "ordinary":
        overlay = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
        config["globals"].update(overlay.pop("globals"))
        config.update(overlay)
    if profile == "schema-entity":
        config["extractor"]["default"]["params"]["schema_path"] = str(paths[2].resolve())
    return config
