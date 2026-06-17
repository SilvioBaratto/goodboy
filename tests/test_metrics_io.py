"""
Source-blind tests for demo/eval/metrics_io.py — Issue #2.

Authored from acceptance criteria only; implementation does not exist yet (Red).

Criteria covered (per oracle report):
  C1 — Module exposes SCHEMA_VERSION, METRICS_DIR, write_metric, read_metric
  C2 — write_metric wraps payload in envelope; creates demo/metrics/ on demand
  C3 — read_metric round-trips payload; envelope keys present; schema_version == 1
  C4 — Module is pure stdlib (no torch/transformers on import)

Criteria skipped (NOT VERIFIABLE per oracle):
  "All tests pass" — boilerplate suite gate, no per-criterion assertion
  "SOLID, clean code …" — subjective prose, no concrete runtime assertion
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from eval import metrics_io as mio


# ---------------------------------------------------------------------------
# C1 — Public surface
# ---------------------------------------------------------------------------


def test_when_imported_then_schema_version_is_exposed():
    assert hasattr(mio, "SCHEMA_VERSION")


def test_when_imported_then_metrics_dir_is_exposed():
    assert hasattr(mio, "METRICS_DIR")


def test_when_imported_then_metrics_dir_leaf_name_is_metrics():
    assert Path(mio.METRICS_DIR).name == "metrics"


def test_when_imported_then_metrics_dir_parent_name_is_demo():
    assert Path(mio.METRICS_DIR).parent.name == "demo"


def test_when_imported_then_write_metric_is_callable():
    assert callable(mio.write_metric)


def test_when_imported_then_read_metric_is_callable():
    assert callable(mio.read_metric)


# ---------------------------------------------------------------------------
# C2 — write_metric: directory creation + envelope shape
# ---------------------------------------------------------------------------


def test_when_write_metric_called_then_metrics_dir_is_created(tmp_path, monkeypatch):
    target = tmp_path / "metrics"
    monkeypatch.setattr(mio, "METRICS_DIR", target)
    mio.write_metric("scores", {"value": 42}, "out.json")
    assert target.is_dir()


def test_when_write_metric_called_then_named_file_exists_in_metrics_dir(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(mio, "METRICS_DIR", tmp_path)
    mio.write_metric("scores", {"value": 42}, "out.json")
    assert (tmp_path / "out.json").exists()


def test_when_write_metric_called_then_envelope_has_schema_version(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(mio, "METRICS_DIR", tmp_path)
    mio.write_metric("scores", {"x": 1}, "out.json")
    data = json.loads((tmp_path / "out.json").read_text())
    assert "schema_version" in data


def test_when_write_metric_called_then_kind_matches_argument(tmp_path, monkeypatch):
    monkeypatch.setattr(mio, "METRICS_DIR", tmp_path)
    mio.write_metric("reward_scores", {"x": 1}, "out.json")
    data = json.loads((tmp_path / "out.json").read_text())
    assert data.get("kind") == "reward_scores"


def test_when_write_metric_called_then_envelope_has_generated_at(tmp_path, monkeypatch):
    monkeypatch.setattr(mio, "METRICS_DIR", tmp_path)
    mio.write_metric("scores", {"x": 1}, "out.json")
    data = json.loads((tmp_path / "out.json").read_text())
    assert "generated_at" in data


def test_when_write_metric_called_then_generated_at_is_iso8601_utc(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(mio, "METRICS_DIR", tmp_path)
    mio.write_metric("scores", {"x": 1}, "out.json")
    data = json.loads((tmp_path / "out.json").read_text())
    ts: str = data["generated_at"]
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 0.0  # type: ignore


def test_when_write_metric_called_then_envelope_has_base_model(tmp_path, monkeypatch):
    monkeypatch.setattr(mio, "METRICS_DIR", tmp_path)
    mio.write_metric("scores", {"x": 1}, "out.json")
    data = json.loads((tmp_path / "out.json").read_text())
    assert "base_model" in data


def test_when_write_metric_called_then_envelope_has_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(mio, "METRICS_DIR", tmp_path)
    mio.write_metric("scores", {"x": 1}, "out.json")
    data = json.loads((tmp_path / "out.json").read_text())
    assert "seed" in data


# ---------------------------------------------------------------------------
# C3 — read_metric round-trip
# ---------------------------------------------------------------------------


def test_when_metric_written_then_read_metric_returns_a_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(mio, "METRICS_DIR", tmp_path)
    mio.write_metric("eval", {"score": 0.9}, "m.json")
    result = mio.read_metric(tmp_path / "m.json")
    assert isinstance(result, dict)


def test_when_metric_written_then_read_metric_schema_version_is_one(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(mio, "METRICS_DIR", tmp_path)
    mio.write_metric("eval", {"score": 0.9}, "m.json")
    result = mio.read_metric(tmp_path / "m.json")
    assert result["schema_version"] == 1


def test_when_metric_written_then_read_metric_has_all_envelope_keys(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(mio, "METRICS_DIR", tmp_path)
    mio.write_metric("eval", {"score": 0.9}, "m.json")
    result = mio.read_metric(tmp_path / "m.json")
    for key in ("schema_version", "kind", "generated_at", "base_model", "seed"):
        assert key in result, f"envelope key missing: {key!r}"


def test_when_metric_written_then_payload_values_survive_round_trip(
    tmp_path, monkeypatch
):
    """Payload values must be recoverable after write_metric -> read_metric.

    Interpretation: payload is accessible either at the top level of the returned
    dict or nested under a 'payload' key — this test accepts either layout.
    Uses non-envelope keys ('loss', 'step') to avoid collision.
    """
    monkeypatch.setattr(mio, "METRICS_DIR", tmp_path)
    payload = {"loss": 0.42, "step": 100}
    mio.write_metric("training", payload, "m.json")
    result = mio.read_metric(tmp_path / "m.json")
    recovered = (
        result.get("payload") if isinstance(result.get("payload"), dict) else result
    )
    assert recovered["loss"] == pytest.approx(0.42)
    assert recovered["step"] == 100


# --- Property: round-trip invariant for any JSON-serialisable payload ---

_ENVELOPE_KEYS = frozenset(
    {"schema_version", "kind", "generated_at", "base_model", "seed", "payload"}
)
_SAFE_KEY = st.text(min_size=1).filter(lambda k: k not in _ENVELOPE_KEYS)
_JSON_LEAF = st.one_of(
    st.text(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
)
_JSON_PAYLOAD = st.dictionaries(_SAFE_KEY, _JSON_LEAF, max_size=8)


@given(payload=_JSON_PAYLOAD)
@settings(max_examples=50, deadline=None)
def test_when_any_json_payload_is_written_then_read_metric_round_trips_it(payload):
    """Round-trip invariant: write_metric then read_metric preserves every value.

    Derived from C3: "read_metric round-trips a written payload."
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        original = mio.METRICS_DIR
        mio.METRICS_DIR = td_path
        try:
            mio.write_metric("prop_test", payload, "round_trip.json")
            result = mio.read_metric(td_path / "round_trip.json")
        finally:
            mio.METRICS_DIR = original

    recovered = (
        result.get("payload") if isinstance(result.get("payload"), dict) else result
    )
    for k, v in payload.items():
        assert recovered[k] == v


# ---------------------------------------------------------------------------
# C4 — Pure stdlib: no torch or transformers loaded on import
# ---------------------------------------------------------------------------

_DEMO_DIR = str(Path(__file__).resolve().parent.parent / "demo")


def test_when_metrics_io_imported_in_fresh_process_then_torch_is_not_loaded():
    """Importing eval.metrics_io in isolation must not pull in torch."""
    code = (
        f"import sys; sys.path.insert(0, {_DEMO_DIR!r}); "
        "import eval.metrics_io; "
        "assert 'torch' not in sys.modules, "
        "'torch was transitively imported by eval.metrics_io'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_when_metrics_io_imported_in_fresh_process_then_transformers_is_not_loaded():
    """Importing eval.metrics_io in isolation must not pull in transformers."""
    code = (
        f"import sys; sys.path.insert(0, {_DEMO_DIR!r}); "
        "import eval.metrics_io; "
        "assert 'transformers' not in sys.modules, "
        "'transformers was transitively imported by eval.metrics_io'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
