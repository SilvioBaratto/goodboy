"""Versioned persistence module for Cycle 1 metric artifacts.

Pure stdlib (json, datetime, pathlib) — safe to import from any script without
pulling in ML dependencies (torch, transformers, etc.).

Callers that know BASE_MODEL / SEED (e.g. weight_delta.py, reward_score.py)
should pass them explicitly so that common.py remains the single source of
truth for those constants; the defaults here are for standalone use and tests.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION: int = 1

_HERE: Path = Path(__file__).resolve().parent   # goodboy/demo/eval/
METRICS_DIR: Path = _HERE.parent / "metrics"    # goodboy/demo/metrics/


def write_metric(
    kind: str,
    payload: dict,
    filename: str,
    *,
    base_model: str = "gpt2",
    seed: int = 42,
    output_dir: "Path | str | None" = None,
) -> Path:
    """Persist *payload* in a versioned envelope to <output_dir>/filename.

    *output_dir* defaults to METRICS_DIR (created on demand). Returns the path
    of the written file.
    """
    target_dir = METRICS_DIR if output_dir is None else Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "seed": seed,
        "payload": payload,
    }
    path = target_dir / filename
    path.write_text(json.dumps(envelope, indent=2))
    return path


def read_metric(path: "Path | str") -> dict:
    """Return the full envelope dict from the JSON file at *path*."""
    return json.loads(Path(path).read_text())
