"""
Source-blind example tests for demo/weight_delta.py (issue #3).

Derived exclusively from the acceptance-criteria text and the oracle report;
no implementation source was read (TDD Red phase — all tests fail until the
implementation exists).

Public API contract defined here:
  compute_param_delta(aligned, base) -> (l2, base_norm, relative)
  combine_group_l2(l2_values)        -> float  (Euclidean combination)
  get_group_name(param_name)         -> str
  validate_param_name_sets(a, b)     -> None  (raises on mismatch)
  build_group_deltas(base, aligned)  -> dict[str, float]  (group -> relative)
  save_weight_delta(per_param, groups, output_dir) -> Path  (versioned JSON + PNG)
  GROUP_ORDER                        -> list[str]  (canonical x-axis order)
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings, strategies as st

from weight_delta import (
    GROUP_ORDER,
    _aggregate_groups,
    _compute_all_params,
    build_group_deltas,
    combine_group_l2,
    compute_param_delta,
    get_group_name,
    save_weight_delta,
    validate_param_name_sets,
)


def _records():
    """Build (per_param, groups) records from a fake base/aligned model pair."""
    base, aligned = _make_model_pair(wte_shift=0.001, block_shift=1.0)
    per_param = _compute_all_params(base, aligned)
    return per_param, _aggregate_groups(per_param)


# ─── test fixtures ────────────────────────────────────────────────────────────


class _FakeModel:
    """Minimal model stand-in: implements named_parameters(); no GPT-2 load."""

    def __init__(self, params: dict) -> None:
        self._params = params

    def named_parameters(self):
        yield from self._params.items()


def _make_model_pair(wte_shift: float, block_shift: float) -> tuple:
    """
    Return (base, aligned) FakeModels with one block.
    wte_shift  — how much the embedding matrix changes.
    block_shift — how much the single attention weight changes.
    """
    base_params = {
        "transformer.wte.weight": torch.ones(10, 4),
        "transformer.wpe.weight": torch.ones(4, 4),
        "transformer.h.0.attn.c_attn.weight": torch.ones(4, 12),
        "transformer.ln_f.weight": torch.ones(4),
    }
    aligned_params = {
        "transformer.wte.weight": torch.ones(10, 4) + wte_shift,
        "transformer.wpe.weight": torch.ones(4, 4) + block_shift * 0.5,
        "transformer.h.0.attn.c_attn.weight": torch.ones(4, 12) + block_shift,
        "transformer.ln_f.weight": torch.ones(4) + block_shift * 0.1,
    }
    return _FakeModel(base_params), _FakeModel(aligned_params)


# ─── Criterion 1: name-set lockstep validation ───────────────────────────────


def test_when_name_sets_differ_then_validation_raises():
    with pytest.raises((AssertionError, ValueError)):
        validate_param_name_sets({"a", "b"}, {"a", "c"})


def test_when_name_sets_match_then_validation_passes():
    names = {"transformer.wte.weight", "transformer.wpe.weight"}
    validate_param_name_sets(names, names)  # must not raise


# ─── Criterion 2: per-param delta computation ────────────────────────────────


def test_when_tensors_differ_then_l2_is_norm_of_difference():
    base = torch.tensor([1.0, 0.0])
    aligned = torch.tensor([4.0, 0.0])  # diff = [3, 0], norm = 3
    l2, _, _ = compute_param_delta(aligned, base)
    assert math.isclose(l2, 3.0, rel_tol=1e-5)


def test_when_tensors_given_then_base_norm_is_norm_of_base_tensor():
    base = torch.tensor([3.0, 4.0])  # norm = 5
    aligned = torch.tensor([3.0, 4.0])
    _, base_norm, _ = compute_param_delta(aligned, base)
    assert math.isclose(base_norm, 5.0, rel_tol=1e-5)


def test_when_tensors_given_then_relative_equals_l2_divided_by_base_norm():
    base = torch.tensor([3.0, 4.0])  # norm = 5
    aligned = torch.tensor([4.0, 4.0])  # diff norm = 1; relative = 0.2
    l2, base_norm, relative = compute_param_delta(aligned, base)
    assert math.isclose(relative, l2 / base_norm, rel_tol=1e-5)


# Criterion 2 property: relative == l2 / base_norm for all positive-valued tensors.
@given(
    a=st.lists(
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=32,
    ),
    b=st.lists(
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=32,
    ),
)
@settings(max_examples=50)
def test_when_any_positive_tensors_given_then_relative_always_equals_l2_over_base_norm(
    a, b
):
    size = max(len(a), len(b))
    t_aligned = torch.tensor((a + [0.1] * size)[:size], dtype=torch.float32)
    t_base = torch.tensor((b + [0.1] * size)[:size], dtype=torch.float32)
    l2, base_norm, relative = compute_param_delta(t_aligned, t_base)
    assert base_norm > 0
    assert math.isclose(relative, l2 / base_norm, rel_tol=1e-4)


# ─── Criterion 3: parameter grouping and Euclidean combination ───────────────


def test_when_param_is_wte_weight_then_group_name_is_transformer_wte():
    assert get_group_name("transformer.wte.weight") == "transformer.wte"


def test_when_param_is_wpe_weight_then_group_name_is_transformer_wpe():
    assert get_group_name("transformer.wpe.weight") == "transformer.wpe"


def test_when_param_is_in_block_0_then_group_name_is_transformer_h_0():
    assert get_group_name("transformer.h.0.attn.c_attn.weight") == "transformer.h.0"


def test_when_param_is_in_block_11_then_group_name_is_transformer_h_11():
    assert get_group_name("transformer.h.11.mlp.c_fc.weight") == "transformer.h.11"


def test_when_param_is_ln_f_weight_then_group_name_is_transformer_ln_f():
    assert get_group_name("transformer.ln_f.weight") == "transformer.ln_f"


def test_when_two_l2_values_given_then_euclidean_combination_is_their_hypotenuse():
    # sqrt(3^2 + 4^2) = 5
    assert math.isclose(combine_group_l2([3.0, 4.0]), 5.0, rel_tol=1e-5)


def test_when_single_l2_value_given_then_combination_returns_that_value():
    assert math.isclose(combine_group_l2([7.0]), 7.0, rel_tol=1e-5)


# Criterion 3 property: Euclidean combination is >= each component (monotonicity).
@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=50)
def test_when_l2_values_combined_then_result_is_at_least_the_largest_component(values):
    result = combine_group_l2(values)
    assert result >= max(values) - 1e-6  # 1e-6 tolerance for float rounding


# ─── Criterion 4: GROUP_ORDER constant and file output ───────────────────────


def test_when_group_order_inspected_then_wte_precedes_wpe():
    assert GROUP_ORDER.index("transformer.wte") < GROUP_ORDER.index("transformer.wpe")


def test_when_group_order_inspected_then_wpe_precedes_first_block():
    assert GROUP_ORDER.index("transformer.wpe") < GROUP_ORDER.index("transformer.h.0")


def test_when_group_order_inspected_then_blocks_are_in_numerically_ascending_order():
    block_positions = [GROUP_ORDER.index(f"transformer.h.{i}") for i in range(12)]
    assert block_positions == sorted(block_positions)


def test_when_group_order_inspected_then_ln_f_is_the_last_entry():
    assert GROUP_ORDER[-1] == "transformer.ln_f"


def test_when_save_weight_delta_called_then_weight_delta_json_is_written(tmp_path):
    per_param, groups = _records()
    save_weight_delta(per_param, groups, tmp_path)
    out = tmp_path / "weight_delta.json"
    assert out.exists(), "weight_delta.json was not written"
    env = json.loads(out.read_text())
    # Versioned envelope (issue #2 contract), not a bare flat dict.
    assert env["schema_version"] == 1
    assert env["kind"] == "weight_delta"
    payload = env["payload"]
    assert {"per_param", "groups", "chart_path"} <= set(payload)
    assert payload["chart_path"] == "weight_delta.png"
    assert payload["groups"], "groups[] must be non-empty"
    for field in ("group", "l2", "base_norm", "relative", "numel"):
        assert field in payload["groups"][0], f"groups[] entry missing '{field}'"
    for field in ("name", "l2", "base_norm", "relative", "numel"):
        assert field in payload["per_param"][0], f"per_param[] entry missing '{field}'"


def test_when_save_weight_delta_called_then_weight_delta_png_is_saved(tmp_path):
    per_param, groups = _records()
    save_weight_delta(per_param, groups, tmp_path)
    assert (tmp_path / "weight_delta.png").exists(), "weight_delta.png was not saved"


# ─── Criterion 5: wte relative << mean block relative (the proof) ────────────


def test_when_wte_barely_moves_and_blocks_move_a_lot_then_wte_relative_is_below_mean_block_relative():
    """
    Constructs a fake model pair where wte barely shifts and the attention
    block shifts significantly, then asserts that build_group_deltas reflects
    the expected ordering (wte_relative < mean_block_relative).
    This pins down the RLHF proof: DPO reshapes block weights, not embeddings.
    """
    base, aligned = _make_model_pair(wte_shift=0.001, block_shift=1.0)
    group_deltas = build_group_deltas(base, aligned)

    wte_relative = group_deltas["transformer.wte"]
    block_relatives = [
        v for k, v in group_deltas.items() if k.startswith("transformer.h.")
    ]
    assert block_relatives, "no transformer block groups found in output"
    mean_block = sum(block_relatives) / len(block_relatives)

    assert wte_relative < mean_block, (
        f"Proof failed: wte_relative={wte_relative:.6f} is not less than "
        f"mean_block_relative={mean_block:.6f}"
    )


# ─── Criterion 6: matplotlib>=3.8 in requirements.txt + headless Agg ────────


def test_when_requirements_txt_is_read_then_matplotlib_gte_3_8_is_declared():
    req = Path("demo/requirements.txt")
    assert req.exists(), "demo/requirements.txt does not exist"
    assert "matplotlib>=3.8" in req.read_text(), (
        "matplotlib>=3.8 not found in demo/requirements.txt"
    )


def test_when_save_weight_delta_called_without_display_then_no_error_is_raised(
    tmp_path,
):
    """
    Criterion 6: weight_delta.py must call matplotlib.use('Agg') before any
    pyplot import so chart creation works on a headless CPU. This test removes
    $DISPLAY from the environment; non-Agg backends raise RuntimeError there.
    """
    old_display = os.environ.pop("DISPLAY", None)
    try:
        per_param, groups = _records()
        save_weight_delta(per_param, groups, tmp_path)  # must not raise RuntimeError
    finally:
        if old_display is not None:
            os.environ["DISPLAY"] = old_display
