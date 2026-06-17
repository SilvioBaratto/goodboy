"""
Source-blind tests for issue #7:
  docs: document RLHF-vs-RLAIF behavior divergence and bias/diversity trade-offs

Oracle-verifiable criteria tested here:

  [UNIT] A standalone demo/RLHF_vs_RLAIF.md documents behavior divergence
         between the human-feedback and AI-feedback models, citing real numbers
         from demo/metrics/rlhf_vs_rlaif.json and the per-model metrics.

  [UNIT] README now HAS the Cycle-3 results section, stage diagrams, and the
         embedded weight-delta chart (these guards were flipped from negative
         to positive when Cycle 3 / issue #10 delivered them).

Skipped (oracle: NOT VERIFIABLE):
  - Explicitly discusses bias/diversity trade-off
  - States the experimental control and toy-overfit caveat
  - All tests pass (boilerplate suite gate)
  - SOLID, clean code, TDD (subjective prose)
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
DEMO_DIR = REPO_ROOT / "demo"
README_PATH = REPO_ROOT / "README.md"
DOC_PATH = DEMO_DIR / "RLHF_vs_RLAIF.md"
METRICS_PATH = DEMO_DIR / "metrics" / "rlhf_vs_rlaif.json"


# ---------------------------------------------------------------------------
# Criterion: demo/RLHF_vs_RLAIF.md exists and cites numbers from metrics JSON
# ---------------------------------------------------------------------------


def test_when_rlhf_vs_rlaif_doc_is_checked_then_file_exists():
    assert DOC_PATH.exists(), (
        f"{DOC_PATH} must exist as the standalone RLHF-vs-RLAIF document"
    )


def test_when_metrics_json_is_checked_then_file_exists():
    # demo/metrics/ is a gitignored, regeneratable build artifact (.gitignore),
    # so on a fresh clone / CI it is legitimately absent. Skip rather than fail
    # when it has not been generated; verify it is a real file when present.
    if not METRICS_PATH.exists():
        pytest.skip(
            f"{METRICS_PATH} not generated yet (gitignored artifact); "
            "run demo/compare_rlhf_rlaif.py to produce it"
        )
    assert METRICS_PATH.is_file()


def test_when_doc_is_read_then_both_rlhf_and_rlaif_are_mentioned():
    assert DOC_PATH.exists(), f"{DOC_PATH} must exist"
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "RLHF" in text, "demo/RLHF_vs_RLAIF.md must mention RLHF"
    assert "RLAIF" in text, "demo/RLHF_vs_RLAIF.md must mention RLAIF"


def test_when_metrics_json_is_parsed_then_at_least_one_numeric_value_appears_in_doc():
    """
    The criterion requires 'citing real numbers from demo/metrics/rlhf_vs_rlaif.json'.
    At least one leaf numeric value from that file must appear verbatim (or rounded to
    1–4 decimal places) inside demo/RLHF_vs_RLAIF.md.

    Choosing the simplest consistent interpretation: a match anywhere in the doc text
    for at least one number from the JSON — not every number, just at least one.
    """
    if not METRICS_PATH.exists():
        pytest.skip(
            f"{METRICS_PATH} not generated yet (gitignored artifact); "
            "run demo/compare_rlhf_rlaif.py to produce it"
        )
    assert DOC_PATH.exists(), f"{DOC_PATH} must exist"

    with METRICS_PATH.open(encoding="utf-8") as fh:
        metrics = json.load(fh)

    doc_text = DOC_PATH.read_text(encoding="utf-8")

    def extract_numbers(obj):
        if isinstance(obj, bool):
            return []
        if isinstance(obj, (int, float)):
            return [obj]
        if isinstance(obj, dict):
            return [n for v in obj.values() for n in extract_numbers(v)]
        if isinstance(obj, list):
            return [n for item in obj for n in extract_numbers(item)]
        return []

    numbers = extract_numbers(metrics)
    assert numbers, f"{METRICS_PATH} must contain at least one numeric value"

    def candidate_strings(v):
        if isinstance(v, int):
            return [str(v)]
        return [str(v), f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", f"{v:.4f}"]

    cited = any(any(s in doc_text for s in candidate_strings(v)) for v in numbers)
    assert cited, (
        "demo/RLHF_vs_RLAIF.md must cite at least one numeric value "
        "from demo/metrics/rlhf_vs_rlaif.json"
    )


# ---------------------------------------------------------------------------
# Criterion (Cycle 3 / issue #10): README.md now HAS the results section and
#            the embedded weight-delta chart — these guards were flipped from
#            negative ("must NOT gain ...") to positive assertions.
# ---------------------------------------------------------------------------


def test_when_readme_is_checked_then_results_section_heading_is_present():
    """
    Feature 9 (results section in README) has been delivered in Cycle 3.
    A heading containing 'Results' must now appear in README.md.
    """
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    assert re.search(r"^#{1,6}\s+Results\b", text, re.MULTILINE | re.IGNORECASE), (
        "README.md must contain a 'Results' section heading"
    )


def test_when_readme_is_checked_then_embedded_image_markup_is_present():
    """
    The weight-delta chart is now embedded in README.md as a markdown image.
    Markdown image syntax (![) must be present.
    """
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    assert "![" in text, (
        "README.md must contain embedded image markdown ![...] for the weight-delta chart"
    )


def test_when_readme_is_checked_then_weight_delta_chart_is_referenced():
    """
    The weight-delta chart artifact path (demo/metrics/weight_delta.png) must
    be referenced in README.md — it is the gitignored generated chart added in Cycle 3.
    Stage diagrams are rendered as ASCII (not mermaid) in this repo.
    """
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    assert "demo/metrics/weight_delta.png" in text, (
        "README.md must reference demo/metrics/weight_delta.png (the weight-delta chart)"
    )
