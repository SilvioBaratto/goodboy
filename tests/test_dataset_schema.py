"""
Tests for dataset schema validation — Issue #8.

Criteria covered:
  C1 — Every row in BOTH files has exactly the keys prompt, chosen, rejected
  C2 — prompt starts with "Question:" and contains the \\nAnswer: marker
  C3 — chosen and rejected each start with exactly one leading space
  C4 — chosen != rejected for every row
  C5 — Schema-checking logic is pure-stdlib (no torch/transformers); subprocess probe
  C6 — Rows are unique by normalized prompt; failure message names the duplicate
  C7 — Row counts pinned: preferences = 1000, seed_examples = 12

Criteria skipped (NOT VERIFIABLE):
  "All tests pass" — boilerplate suite gate, no per-criterion assertion
  "SOLID, clean code ..." — subjective prose, no concrete runtime assertion
"""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFERENCES_PATH = REPO_ROOT / "demo" / "data" / "preferences.jsonl"
SEED_PATH = REPO_ROOT / "demo" / "data" / "seed_examples.jsonl"
_DEMO_DIR = str(REPO_ROOT / "demo")

_BOTH_FILES = [
    pytest.param(PREFERENCES_PATH, id="preferences_1000"),
    pytest.param(SEED_PATH, id="seed_examples_12"),
]

_REQUIRED_KEYS = {"prompt", "chosen", "rejected"}


def _load_rows(path: Path) -> list[dict]:
    """Return parsed dicts for every non-blank line in a JSONL file."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _exactly_one_leading_space(s: str) -> bool:
    """True iff s starts with exactly one space character."""
    return s.startswith(" ") and not s.startswith("  ")


# ---------------------------------------------------------------------------
# C1 — Every row has exactly the keys: prompt, chosen, rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_path", _BOTH_FILES)
def test_when_rows_loaded_then_every_row_has_exactly_required_keys(data_path):
    rows = _load_rows(data_path)
    offending = [
        (i, set(r.keys()))
        for i, r in enumerate(rows)
        if set(r.keys()) != _REQUIRED_KEYS
    ]
    assert not offending, (
        f"Rows with unexpected key sets in {data_path.name}: {offending[:5]}"
    )


# ---------------------------------------------------------------------------
# C2a — prompt starts with "Question:"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_path", _BOTH_FILES)
def test_when_rows_loaded_then_every_prompt_starts_with_question_colon(data_path):
    rows = _load_rows(data_path)
    offending = [
        (i, repr(r["prompt"][:60]))
        for i, r in enumerate(rows)
        if not r["prompt"].startswith("Question:")
    ]
    assert not offending, (
        f"Prompts not starting with 'Question:' in {data_path.name}: {offending[:5]}"
    )


# ---------------------------------------------------------------------------
# C2b — prompt contains the \nAnswer: marker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_path", _BOTH_FILES)
def test_when_rows_loaded_then_every_prompt_contains_newline_answer_marker(data_path):
    rows = _load_rows(data_path)
    offending = [
        (i, repr(r["prompt"][:60]))
        for i, r in enumerate(rows)
        if "\nAnswer:" not in r["prompt"]
    ]
    assert not offending, (
        f"Prompts missing '\\nAnswer:' marker in {data_path.name}: {offending[:5]}"
    )


# ---------------------------------------------------------------------------
# C3a — chosen starts with exactly one leading space
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_path", _BOTH_FILES)
def test_when_rows_loaded_then_every_chosen_starts_with_exactly_one_space(data_path):
    rows = _load_rows(data_path)
    offending = [
        (i, repr(r["chosen"][:30]))
        for i, r in enumerate(rows)
        if not _exactly_one_leading_space(r["chosen"])
    ]
    assert not offending, (
        f"chosen completions without exactly one leading space in {data_path.name}: "
        f"{offending[:5]}"
    )


# ---------------------------------------------------------------------------
# C3b — rejected starts with exactly one leading space
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_path", _BOTH_FILES)
def test_when_rows_loaded_then_every_rejected_starts_with_exactly_one_space(data_path):
    rows = _load_rows(data_path)
    offending = [
        (i, repr(r["rejected"][:30]))
        for i, r in enumerate(rows)
        if not _exactly_one_leading_space(r["rejected"])
    ]
    assert not offending, (
        f"rejected completions without exactly one leading space in {data_path.name}: "
        f"{offending[:5]}"
    )


# ---------------------------------------------------------------------------
# C3 properties — "exactly one leading space" is a crisp predicate; its
# boundary behaviour (zero spaces → False, two+ spaces → False) is an
# invariant over all non-space-prefixed strings.
# ---------------------------------------------------------------------------


@given(st.text(min_size=1).filter(lambda s: not s.startswith(" ")))
@settings(max_examples=200, deadline=None)
def test_when_completion_has_no_leading_space_then_exactly_one_check_is_false(body):
    """Criterion: reject zero leading spaces. Any non-space-prefixed body must fail."""
    assert not _exactly_one_leading_space(body)


@given(st.text(min_size=1).filter(lambda s: not s.startswith(" ")))
@settings(max_examples=200, deadline=None)
def test_when_completion_has_two_or_more_leading_spaces_then_check_is_false(body):
    """Criterion: reject >=2 leading spaces. Two-space prefix must always fail."""
    assert not _exactly_one_leading_space("  " + body)


@given(st.text(min_size=1).filter(lambda s: not s.startswith(" ")))
@settings(max_examples=200, deadline=None)
def test_when_completion_has_exactly_one_leading_space_then_check_is_true(body):
    """Criterion: accept exactly one leading space. Single-space prefix must always pass."""
    assert _exactly_one_leading_space(" " + body)


# ---------------------------------------------------------------------------
# C4 — chosen != rejected for every row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_path", _BOTH_FILES)
def test_when_rows_loaded_then_chosen_and_rejected_differ_for_every_row(data_path):
    rows = _load_rows(data_path)
    offending = [
        (i, repr(r["chosen"][:40]))
        for i, r in enumerate(rows)
        if r["chosen"] == r["rejected"]
    ]
    assert not offending, (
        f"Rows with identical chosen/rejected in {data_path.name}: {offending[:5]}"
    )


# ---------------------------------------------------------------------------
# C6 — Rows are unique by normalized prompt; failure message names the duplicate
# ---------------------------------------------------------------------------


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip()).lower()


@pytest.mark.parametrize("data_path", _BOTH_FILES)
def test_when_rows_loaded_then_prompts_are_unique_by_normalized_form(data_path):
    rows = _load_rows(data_path)
    normalized = [_normalize_prompt(r["prompt"]) for r in rows]
    counts = Counter(normalized)
    duplicates = {prompt: n for prompt, n in counts.items() if n > 1}
    assert not duplicates, (
        f"Duplicate normalized prompts in {data_path.name}: "
        + "; ".join(f"{p!r} (appears {n}x)" for p, n in list(duplicates.items())[:3])
    )


# ---------------------------------------------------------------------------
# C7 — Row counts pinned: preferences = 1000, seed_examples = 12
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data_path,expected_count",
    [
        pytest.param(PREFERENCES_PATH, 1000, id="preferences_1000"),
        pytest.param(SEED_PATH, 12, id="seed_examples_12"),
    ],
)
def test_when_rows_loaded_then_row_count_matches_pinned_value(
    data_path, expected_count
):
    rows = _load_rows(data_path)
    assert len(rows) == expected_count, (
        f"{data_path.name}: expected {expected_count} rows, got {len(rows)}"
    )


# ---------------------------------------------------------------------------
# C5 — Schema-checking logic uses only stdlib (json, pathlib) — no torch/
#       transformers loaded as a side-effect of running the validation.
# ---------------------------------------------------------------------------


def test_when_schema_validation_runs_then_torch_is_not_imported():
    """Performing JSONL schema checks must not pull in torch."""
    code = (
        "import sys, json, pathlib; "
        f"path = pathlib.Path({str(PREFERENCES_PATH)!r}); "
        "rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]; "
        "_ = [set(r.keys()) == {'prompt','chosen','rejected'} for r in rows]; "
        "assert 'torch' not in sys.modules, 'torch was transitively imported during schema validation'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_when_schema_validation_runs_then_transformers_is_not_imported():
    """Performing JSONL schema checks must not pull in transformers."""
    code = (
        "import sys, json, pathlib; "
        f"path = pathlib.Path({str(PREFERENCES_PATH)!r}); "
        "rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]; "
        "_ = [set(r.keys()) == {'prompt','chosen','rejected'} for r in rows]; "
        "assert 'transformers' not in sys.modules, "
        "'transformers was transitively imported during schema validation'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
