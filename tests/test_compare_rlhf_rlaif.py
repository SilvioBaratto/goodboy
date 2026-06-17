"""Source-blind tests for Issue #6: RLHF-vs-RLAIF comparison driver.

Authored from acceptance criteria only (no implementation source read).
TDD seam: module-level path constants and the pure build_report() function
allow full unit coverage without ever touching the GPU or reward model.

Design decisions recorded here (derived from criteria, not source):
  - compare_rlhf_rlaif exposes RLHF_METRICS_DIR and RLAIF_METRICS_DIR constants.
  - compare_rlhf_rlaif exposes a pure build_report(rlhf_reward_delta,
    rlaif_reward_delta, diversity_proxy, per_prompt_texts) -> dict function.
  - compare_rlhf_rlaif exposes COMBINED_OUTPUT_FILE naming the output JSON.
  - Reuse is verified by confirming that Scorer, save_weight_delta, and
    write_metrics appear in the module namespace (imported, not re-coded).
"""

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, strategies as st


# ── Criterion 1: eval/common.py gains ALIGNED_SEED + load_model_from (additive) ──


def test_when_common_imported_then_ALIGNED_SEED_points_to_aligned_seed_dir():
    """ALIGNED_SEED is a Path to the RLHF checkpoint, mirroring ALIGNED for RLAIF.

    The source-blind test assumed it was an integer seed; the issue body explicitly
    defines it as `HERE.parent / "aligned_seed"` (a Path constant).
    """
    from eval import common  # type: ignore

    p = Path(common.ALIGNED_SEED)
    assert p.name == "aligned_seed", f"Expected name 'aligned_seed', got: {p.name}"
    assert p.parent.name == "demo", f"Expected parent 'demo', got: {p.parent.name}"


def test_when_common_imported_then_load_model_from_is_callable():
    from eval import common  # type: ignore

    assert callable(common.load_model_from), (
        "eval/common.py must expose load_model_from as a callable"
    )


def test_when_load_model_from_inspected_then_it_accepts_a_path_argument():
    """load_model_from is path-parametric; the spec requires it takes a path."""
    from eval import common  # type: ignore

    sig = inspect.signature(common.load_model_from)
    assert len(sig.parameters) >= 1, (
        f"load_model_from must accept at least one argument (the path); "
        f"found {len(sig.parameters)} parameters"
    )


def test_when_common_imported_then_load_aligned_still_exists():
    """Additive rule: load_aligned() must not be removed by this change."""
    from eval import common  # type: ignore

    assert callable(common.load_aligned), (
        "load_aligned() must still exist in eval/common.py after the additive change"
    )


def test_when_load_model_from_called_with_existing_path_then_eval_is_invoked(tmp_path):
    """load_model_from must call .eval() on the loaded model, like load_aligned."""
    from eval import common  # type: ignore

    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    with patch(
        "eval.common.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    ):
        common.load_model_from(tmp_path)
    mock_model.eval.assert_called_once()


def test_when_load_model_from_called_with_missing_path_then_system_exit_is_raised(
    tmp_path,
):
    """load_model_from must raise SystemExit for a missing path, like load_aligned."""
    from eval import common  # type: ignore

    missing = tmp_path / "nonexistent_model"
    with pytest.raises(SystemExit):
        common.load_model_from(missing)


# ── _diversity tests (from issue comment: concrete deterministic diversity metric) ──


def test_when_diversity_called_with_identical_prompts_then_distinct_ratio_is_reciprocal_of_n():
    from compare_rlhf_rlaif import _diversity  # type: ignore

    pairs = [
        {"prompt": "same prompt", "chosen": "a", "rejected": "b"} for _ in range(4)
    ]
    result = _diversity(pairs)
    assert abs(result["distinct_prompt_ratio"] - 0.25) < 1e-9


def test_when_diversity_called_with_all_different_prompts_then_distinct_ratio_is_1():
    from compare_rlhf_rlaif import _diversity  # type: ignore

    pairs = [
        {"prompt": f"unique {i}", "chosen": "a", "rejected": "b"} for i in range(5)
    ]
    result = _diversity(pairs)
    assert result["distinct_prompt_ratio"] == 1.0


def test_when_diversity_called_then_result_has_required_keys():
    from compare_rlhf_rlaif import _diversity  # type: ignore

    result = _diversity([{"prompt": "Q1", "chosen": "a", "rejected": "b"}])
    assert "distinct_prompt_ratio" in result
    assert "mean_pairwise_jaccard" in result


def test_when_diversity_called_with_identical_prompts_then_jaccard_is_1():
    from compare_rlhf_rlaif import _diversity  # type: ignore

    pairs = [
        {"prompt": "same tokens here", "chosen": "a", "rejected": "b"} for _ in range(3)
    ]
    result = _diversity(pairs)
    assert abs(result["mean_pairwise_jaccard"] - 1.0) < 1e-9


def test_when_diversity_called_with_empty_pairs_then_zeros_are_returned():
    from compare_rlhf_rlaif import _diversity  # type: ignore

    result = _diversity([])
    assert result["distinct_prompt_ratio"] == 0.0
    assert result["mean_pairwise_jaccard"] == 0.0


# ── Criterion 2: HELD_OUT_PROMPTS + common.generate (same seed + greedy) ──────


def test_when_common_imported_then_HELD_OUT_PROMPTS_is_a_nonempty_sequence():
    from eval import common  # type: ignore

    prompts = common.HELD_OUT_PROMPTS
    assert len(list(prompts)) > 0, (
        "HELD_OUT_PROMPTS must be a non-empty sequence; got an empty collection"
    )


def test_when_compare_module_imported_then_a_run_or_main_callable_is_exposed():
    """The driver script must have an entry point callable (run or main)."""
    import compare_rlhf_rlaif  # type: ignore

    entry = getattr(compare_rlhf_rlaif, "run", None) or getattr(
        compare_rlhf_rlaif, "main", None
    )
    assert callable(entry), (
        "compare_rlhf_rlaif must expose run() or main() as its entry point"
    )


# ── Criterion 3: Scorer (once) + metric math not reimplemented ────────────────


def test_when_compare_module_imported_then_Scorer_is_in_namespace():
    """Scorer must be imported from eval tooling, not reimplemented.

    If 'Scorer' appears in the module namespace it was imported (reused).
    A reimplementation would appear under a different name or inside the module
    body, but not as a top-level name that matches the eval Scorer contract.
    """
    import compare_rlhf_rlaif  # type: ignore

    assert "Scorer" in dir(compare_rlhf_rlaif), (
        "compare_rlhf_rlaif must import Scorer from eval tooling; "
        "'Scorer' was not found in the module namespace"
    )


def test_when_compare_module_imported_then_save_weight_delta_is_in_namespace():
    """save_weight_delta must be imported from eval tooling (zero reimplementation)."""
    import compare_rlhf_rlaif  # type: ignore

    assert "save_weight_delta" in dir(compare_rlhf_rlaif), (
        "compare_rlhf_rlaif must import save_weight_delta from eval tooling"
    )


def test_when_compare_module_imported_then_write_metrics_is_in_namespace():
    """write_metrics must be imported from eval tooling (zero reimplementation)."""
    import compare_rlhf_rlaif  # type: ignore

    assert "write_metrics" in dir(compare_rlhf_rlaif), (
        "compare_rlhf_rlaif must import write_metrics from eval tooling"
    )


# ── Criterion 4: RLHF → metrics/aligned_seed/; never clobber RLAIF metrics/ ──


def test_when_compare_module_imported_then_rlhf_metrics_dir_ends_in_aligned_seed():
    import compare_rlhf_rlaif  # type: ignore

    rlhf_dir = Path(compare_rlhf_rlaif.RLHF_METRICS_DIR)
    assert rlhf_dir.name == "aligned_seed", (
        f"RLHF metrics must route to .../metrics/aligned_seed/, "
        f"but RLHF_METRICS_DIR ends in: '{rlhf_dir.name}'"
    )


def test_when_compare_module_imported_then_rlhf_metrics_dir_contains_metrics_segment():
    import compare_rlhf_rlaif  # type: ignore

    rlhf_dir = Path(compare_rlhf_rlaif.RLHF_METRICS_DIR)
    assert "metrics" in rlhf_dir.parts, (
        f"Expected 'metrics' to appear in RLHF_METRICS_DIR path; got: {rlhf_dir}"
    )


def test_when_rlhf_and_rlaif_metrics_dirs_compared_then_they_are_different_paths():
    import compare_rlhf_rlaif  # type: ignore

    rlhf_dir = Path(compare_rlhf_rlaif.RLHF_METRICS_DIR)
    rlaif_dir = Path(compare_rlhf_rlaif.RLAIF_METRICS_DIR)
    assert rlhf_dir != rlaif_dir, (
        f"RLHF and RLAIF metrics dirs must be distinct to avoid clobbering; "
        f"both resolved to: {rlhf_dir}"
    )


def test_when_rlhf_metrics_dir_checked_then_it_is_not_the_bare_metrics_dir():
    """RLHF must NOT write to bare metrics/ — that directory belongs to RLAIF."""
    import compare_rlhf_rlaif  # type: ignore

    rlhf_dir = Path(compare_rlhf_rlaif.RLHF_METRICS_DIR)
    assert rlhf_dir.name != "metrics", (
        "RLHF_METRICS_DIR must end in a subdirectory (aligned_seed/), "
        "not the bare metrics/ root that RLAIF uses"
    )


# ── Criterion 5: rlhf_vs_rlaif.json via metrics_io.write_metric ───────────────


def test_when_compare_module_imported_then_combined_output_filename_is_rlhf_vs_rlaif_json():
    """The script must advertise the exact combined output filename as a constant."""
    import compare_rlhf_rlaif  # type: ignore

    filename_const = getattr(
        compare_rlhf_rlaif, "COMBINED_OUTPUT_FILE", None
    ) or getattr(compare_rlhf_rlaif, "COMPARISON_FILE", None)
    assert filename_const is not None, (
        "compare_rlhf_rlaif must expose the combined output filename "
        "(COMBINED_OUTPUT_FILE or COMPARISON_FILE) as a module constant"
    )
    assert Path(str(filename_const)).name == "rlhf_vs_rlaif.json", (
        f"Combined output filename must be 'rlhf_vs_rlaif.json'; "
        f"got: '{Path(str(filename_const)).name}'"
    )


def test_when_build_report_called_then_result_contains_reward_deltas():
    """The combined report must compare reward deltas for RLHF vs RLAIF.

    Design decision: build_report(rlhf_reward_delta, rlaif_reward_delta,
    diversity_proxy, per_prompt_texts) -> dict is the pure assembly function.
    """
    from compare_rlhf_rlaif import build_report  # type: ignore

    report = build_report(
        rlhf_reward_delta=0.12,
        rlaif_reward_delta=0.23,
        diversity_proxy=0.75,
        per_prompt_texts=[
            {"prompt": "Question: X\nAnswer:", "rlhf": " calm", "rlaif": " calmer"}
        ],
    )
    assert "reward_deltas" in report, (
        f"report must contain 'reward_deltas'; got keys: {list(report)}"
    )


def test_when_build_report_called_then_result_contains_diversity_proxy():
    from compare_rlhf_rlaif import build_report  # type: ignore

    report = build_report(
        rlhf_reward_delta=0.12,
        rlaif_reward_delta=0.23,
        diversity_proxy=0.75,
        per_prompt_texts=[
            {"prompt": "Question: X\nAnswer:", "rlhf": " calm", "rlaif": " calmer"}
        ],
    )
    assert "diversity_proxy" in report, (
        f"report must contain 'diversity_proxy'; got keys: {list(report)}"
    )


def test_when_build_report_called_then_result_contains_per_prompt_aligned_texts():
    from compare_rlhf_rlaif import build_report  # type: ignore

    report = build_report(
        rlhf_reward_delta=0.12,
        rlaif_reward_delta=0.23,
        diversity_proxy=0.75,
        per_prompt_texts=[
            {"prompt": "Question: X\nAnswer:", "rlhf": " calm", "rlaif": " calmer"}
        ],
    )
    text_key = next(
        (k for k in report if "text" in k.lower() or "prompt" in k.lower()),
        None,
    )
    assert text_key is not None, (
        f"report must contain a key for per-prompt aligned texts "
        f"(e.g. 'per_prompt_texts'); got keys: {list(report)}"
    )


def test_when_compare_module_imported_then_write_metric_is_in_namespace():
    """metrics_io.write_metric must be imported (versioned envelope, not inline JSON dump)."""
    import compare_rlhf_rlaif  # type: ignore

    assert "write_metric" in dir(compare_rlhf_rlaif), (
        "compare_rlhf_rlaif must import write_metric from metrics_io "
        "(versioned envelope); 'write_metric' was not found in the module namespace"
    )


# ── Property: build_report always produces required keys for any valid floats ──


@given(
    rlhf_reward_delta=st.floats(allow_nan=False, allow_infinity=False),
    rlaif_reward_delta=st.floats(allow_nan=False, allow_infinity=False),
    diversity_proxy=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
def test_when_build_report_called_with_any_valid_floats_then_required_keys_are_always_present(
    rlhf_reward_delta: float,
    rlaif_reward_delta: float,
    diversity_proxy: float,
) -> None:
    """Invariant: for any finite float inputs the combined report always contains
    all three required keys (reward_deltas, diversity_proxy, and a texts key).

    This pins the structural contract: build_report is a total function over its
    stated domain and never silently drops a required field.
    """
    from compare_rlhf_rlaif import build_report  # type: ignore

    report = build_report(
        rlhf_reward_delta=rlhf_reward_delta,
        rlaif_reward_delta=rlaif_reward_delta,
        diversity_proxy=diversity_proxy,
        per_prompt_texts=[],
    )

    assert "reward_deltas" in report, (
        f"reward_deltas missing for inputs "
        f"({rlhf_reward_delta}, {rlaif_reward_delta}, {diversity_proxy})"
    )
    assert "diversity_proxy" in report, (
        f"diversity_proxy missing for inputs "
        f"({rlhf_reward_delta}, {rlaif_reward_delta}, {diversity_proxy})"
    )
    text_key = next(
        (k for k in report if "text" in k.lower() or "prompt" in k.lower()),
        None,
    )
    assert text_key is not None, (
        f"per-prompt texts key missing for inputs "
        f"({rlhf_reward_delta}, {rlaif_reward_delta}, {diversity_proxy})"
    )
