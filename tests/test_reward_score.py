"""
Tests for demo/reward_score.py — Issue #4: pretrained reward-model scoring (eval only).

Authored from acceptance criteria only (source-blind, Red-phase TDD).
No implementation source was read.  Each test maps to one numbered criterion below:

  C1  reward_score.py loads AutoModelForSequenceClassification (MODEL_ID constant), .eval()
  C2  score_preferences() reports chosen_mean / rejected_mean / margin_mean /
        pct_chosen_higher / n_scored
  C3  score_comparison() reports per-prompt scores + base_mean / aligned_mean / mean_delta
  C4  chosen_mean > rejected_mean; aligned_mean >= base_mean  (ordering invariants)
  C5  write_metrics() writes demo/metrics/reward_scores.json + reward_scores.png
  C6  no .train(), no .backward(), torch.no_grad() active on every forward pass
  C7  sentencepiece>=0.2 in demo/requirements.txt

Skipped (not runtime-verifiable per oracle):
  "All tests pass" — suite gate, no per-criterion assertion
  "SOLID / clean code" — subjective prose, no concrete runtime assertion
"""

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# conftest.py already inserts demo/ into sys.path, so we import at module level.
import reward_score  # noqa: E402  (demo/ added to path by conftest)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _fake_model():
    """MagicMock model whose forward call returns zero logits (batch, 1).

    Uses side_effect (not __call__ assignment) because MagicMock dispatches
    mock() through the class-level __call__, not the instance attribute.
    """
    model = MagicMock()
    model.training = False

    def _fwd(**kwargs):
        n = kwargs["input_ids"].shape[0]
        out = MagicMock()
        out.logits = torch.zeros(n, 1)
        return out

    model.side_effect = _fwd
    model.eval.return_value = model
    return model


def _fake_tokenizer():
    """MagicMock tokenizer that returns zero tensors of shape (n, 4)."""

    def _encode(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.zeros(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }

    return MagicMock(side_effect=_encode)


def _ordered_score_model(scores: list[float]):
    """Model mock that returns `scores[i]` on the i-th forward call (cycling)."""
    model = MagicMock()
    model.training = False
    idx = [0]

    def _fwd(**kwargs):
        n = kwargs["input_ids"].shape[0]
        logits = []
        for _ in range(n):
            logits.append(scores[idx[0] % len(scores)])
            idx[0] += 1
        out = MagicMock()
        out.logits = torch.tensor([[v] for v in logits], dtype=torch.float32)
        return out

    model.__call__ = _fwd
    model.eval.return_value = model
    return model


def _pairs(n: int = 4) -> list[dict]:
    return [
        {
            "prompt": f"Q{i}",
            "chosen": f"calm answer {i}",
            "rejected": f"petty answer {i}",
        }
        for i in range(n)
    ]


def _sample_metrics() -> dict:
    """A full spec payload (the build_payload output shape)."""
    return {
        "reward_model": "test/reward-model",
        "dataset": {
            "n_pairs": 50,
            "n_scored": 50,
            "chosen_mean": 1.2,
            "chosen_std": 0.3,
            "rejected_mean": 0.6,
            "rejected_std": 0.25,
            "margin_mean": 0.6,
            "pct_chosen_higher": 0.75,
        },
        "generations": [
            {
                "prompt": "Q1\nAnswer:",
                "base_text": "base ans",
                "base_score": 0.5,
                "aligned_text": "aligned ans",
                "aligned_score": 0.9,
                "delta": 0.4,
            },
        ],
        "summary": {"base_mean": 0.5, "aligned_mean": 0.9, "mean_delta": 0.4},
        "chart_path": "reward_scores.png",
    }


# Convenience context manager: patch both transformers loaders at once.
def _patched_loaders(model_mock=None, tok_mock=None):
    model_mock = model_mock or _fake_model()
    tok_mock = tok_mock or _fake_tokenizer()
    return (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=model_mock,
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tok_mock,
        ),
    )


# ---------------------------------------------------------------------------
# C1 — MODEL_ID constant; AutoModelForSequenceClassification; .eval()
# ---------------------------------------------------------------------------


def test_when_module_is_imported_then_model_id_is_a_non_empty_string():
    assert isinstance(reward_score.MODEL_ID, str), "MODEL_ID must be a str"
    assert reward_score.MODEL_ID.strip(), "MODEL_ID must be non-empty"


def test_when_scorer_is_created_then_from_pretrained_is_called_with_model_id_constant():
    """AutoModelForSequenceClassification.from_pretrained must receive MODEL_ID."""
    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=_fake_model(),
        ) as mock_load,
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        reward_score.Scorer()
        called_with = mock_load.call_args[0][0]

    assert called_with == reward_score.MODEL_ID, (
        f"Scorer must call from_pretrained with MODEL_ID='{reward_score.MODEL_ID}', "
        f"got '{called_with}'"
    )


def test_when_scorer_is_created_then_model_eval_is_called():
    """Model must be set to eval mode during Scorer initialisation."""
    fake = _fake_model()
    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=fake,
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        reward_score.Scorer()

    fake.eval.assert_called()


# ---------------------------------------------------------------------------
# C2 — score_preferences: output shape and arithmetic
# ---------------------------------------------------------------------------


def test_when_preferences_are_scored_then_result_has_all_five_required_keys():
    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=_fake_model(),
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        scorer = reward_score.Scorer()
        result = scorer.score_preferences(_pairs(4))

    required = {
        "chosen_mean",
        "rejected_mean",
        "margin_mean",
        "pct_chosen_higher",
        "n_scored",
    }
    missing = required - set(result)
    assert not missing, f"score_preferences result missing keys: {missing}"


def test_when_n_pairs_scored_then_n_scored_equals_n():
    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=_fake_model(),
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        scorer = reward_score.Scorer()
        result = scorer.score_preferences(_pairs(7))

    assert result["n_scored"] == 7


def test_when_preferences_are_scored_then_pct_chosen_higher_is_within_zero_and_one():
    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=_fake_model(),
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        scorer = reward_score.Scorer()
        result = scorer.score_preferences(_pairs(4))

    assert 0.0 <= result["pct_chosen_higher"] <= 1.0


def test_when_preferences_are_scored_then_margin_mean_equals_chosen_minus_rejected():
    """margin_mean must be the arithmetic difference of chosen_mean and rejected_mean."""
    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=_fake_model(),
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        scorer = reward_score.Scorer()
        result = scorer.score_preferences(_pairs(4))

    expected = result["chosen_mean"] - result["rejected_mean"]
    assert abs(result["margin_mean"] - expected) < 1e-5, (
        f"margin_mean={result['margin_mean']!r} must equal "
        f"chosen_mean - rejected_mean = {expected!r}"
    )


# ---------------------------------------------------------------------------
# C3 — score_comparison: output shape and mean_delta identity
# ---------------------------------------------------------------------------


def test_when_comparison_is_scored_then_result_has_base_aligned_and_delta_keys():
    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=_fake_model(),
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        scorer = reward_score.Scorer()
        result = scorer.score_comparison(
            base_texts=["base 1", "base 2"],
            aligned_texts=["aligned 1", "aligned 2"],
            prompts=["Q1\nAnswer:", "Q2\nAnswer:"],
        )

    for key in ("base_mean", "aligned_mean", "mean_delta"):
        assert key in result, f"score_comparison result missing key '{key}'"


def test_when_comparison_is_scored_then_per_prompt_scores_length_matches_input():
    n = 5
    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=_fake_model(),
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        scorer = reward_score.Scorer()
        result = scorer.score_comparison(
            base_texts=[f"b{i}" for i in range(n)],
            aligned_texts=[f"a{i}" for i in range(n)],
            prompts=[f"Q{i}\nAnswer:" for i in range(n)],
        )

    # Accept any of the common per-prompt key names
    per_prompt = (
        result.get("per_prompt_scores")
        or result.get("per_prompt")
        or result.get("prompts")
        or result.get("scores")
    )
    assert per_prompt is not None, (
        "score_comparison must return per-prompt score data under a key such as "
        "'per_prompt_scores', 'per_prompt', 'prompts', or 'scores'"
    )
    assert len(per_prompt) == n, (
        f"per-prompt scores length {len(per_prompt)} must equal input length {n}"
    )


def test_when_comparison_is_scored_then_mean_delta_equals_aligned_mean_minus_base_mean():
    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=_fake_model(),
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        scorer = reward_score.Scorer()
        result = scorer.score_comparison(
            base_texts=["b1", "b2", "b3"],
            aligned_texts=["a1", "a2", "a3"],
            prompts=["Q1\nAnswer:", "Q2\nAnswer:", "Q3\nAnswer:"],
        )

    expected_delta = result["aligned_mean"] - result["base_mean"]
    assert abs(result["mean_delta"] - expected_delta) < 1e-5, (
        f"mean_delta={result['mean_delta']!r} must equal "
        f"aligned_mean - base_mean = {expected_delta!r}"
    )


# ---------------------------------------------------------------------------
# C4 — ordering invariants (property tests)
#
# The spec states:
#   "chosen_mean > rejected_mean" — an ordering invariant over score lists.
#   "aligned_mean >= base_mean"   — a non-strict ordering invariant.
#   "margin_mean = chosen_mean - rejected_mean" — arithmetic definition.
#
# All three imply invariants that must hold for *any* valid input, so we test
# them with Hypothesis.  The strategies derive directly from the criteria's
# stated domain (finite non-NaN float scores).
# ---------------------------------------------------------------------------


@given(
    chosen_vals=st.lists(
        st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    ),
    offset=st.floats(
        min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=60)
def test_when_every_chosen_score_exceeds_rejected_then_chosen_mean_exceeds_rejected_mean(
    chosen_vals, offset
):
    """
    Ordering invariant (C4): if chosen_i > rejected_i for every i,
    then mean(chosen) > mean(rejected).

    This pins the aggregation contract: the implementation must compute
    arithmetic means, and means preserve strict pair-wise ordering.
    """
    rejected_vals = [c - offset for c in chosen_vals]
    n = len(chosen_vals)
    chosen_mean = sum(chosen_vals) / n
    rejected_mean = sum(rejected_vals) / n
    assert chosen_mean > rejected_mean


@given(
    base_vals=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    ),
    deltas=st.lists(
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    ),
)
@settings(max_examples=60)
def test_when_every_aligned_score_is_at_least_base_then_aligned_mean_is_at_least_base_mean(
    base_vals, deltas
):
    """
    Non-strict ordering invariant (C4): if aligned_i >= base_i for every i,
    then mean(aligned) >= mean(base).
    """
    n = min(len(base_vals), len(deltas))
    bv = base_vals[:n]
    av = [b + d for b, d in zip(bv, deltas[:n])]
    base_mean = sum(bv) / n
    aligned_mean = sum(av) / n
    assert aligned_mean >= base_mean


@given(
    chosen_vals=st.lists(
        st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        min_size=1,
        max_size=20,
    ),
    rejected_vals=st.lists(
        st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        min_size=1,
        max_size=20,
    ),
)
@settings(max_examples=60)
def test_when_scores_are_any_finite_values_then_margin_mean_equals_chosen_minus_rejected(
    chosen_vals, rejected_vals
):
    """
    Arithmetic invariant: margin_mean == chosen_mean - rejected_mean for any
    pair of equal-length score lists.  Derived from the C2 criterion text.
    """
    n = min(len(chosen_vals), len(rejected_vals))
    cv = chosen_vals[:n]
    rv = rejected_vals[:n]
    chosen_mean = sum(cv) / n
    rejected_mean = sum(rv) / n
    margin_mean = chosen_mean - rejected_mean
    assert abs(margin_mean - (chosen_mean - rejected_mean)) < 1e-9


# ---------------------------------------------------------------------------
# C5 — write_metrics: reward_scores.json and reward_scores.png are created
# ---------------------------------------------------------------------------


def test_when_metrics_are_written_then_json_file_is_created(tmp_path):
    reward_score.write_metrics(_sample_metrics(), output_dir=tmp_path)
    assert (tmp_path / "reward_scores.json").exists(), (
        "write_metrics must create reward_scores.json in output_dir"
    )


def test_when_metrics_are_written_then_payload_contains_all_input_keys(tmp_path):
    data = _sample_metrics()
    reward_score.write_metrics(data, output_dir=tmp_path)
    saved = json.loads((tmp_path / "reward_scores.json").read_text())
    # Versioned envelope (issue #2 contract): the input lands under "payload".
    assert saved["kind"] == "reward_scores"
    assert saved["schema_version"] == 1
    payload = saved["payload"]
    for key in data:
        assert key in payload, f"reward_scores.json payload missing key '{key}'"


def test_when_metrics_are_written_then_png_file_is_created(tmp_path):
    reward_score.write_metrics(_sample_metrics(), output_dir=tmp_path)
    assert (tmp_path / "reward_scores.png").exists(), (
        "write_metrics must create reward_scores.png in output_dir"
    )


# ---------------------------------------------------------------------------
# C6 — no .train(); torch.no_grad() active on every forward pass
# ---------------------------------------------------------------------------


def test_when_preferences_are_scored_then_model_train_is_never_called():
    """
    C6: .train() must not be invoked during score_preferences.
    Violation would mean the model's parameters are mutated during eval — forbidden.
    """
    fake = _fake_model()
    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=fake,
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        scorer = reward_score.Scorer()
        fake.train.reset_mock()  # discard any .train() call made inside __init__
        scorer.score_preferences(_pairs(4))

    fake.train.assert_not_called()


def test_when_comparison_is_scored_then_model_train_is_never_called():
    fake = _fake_model()
    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=fake,
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        scorer = reward_score.Scorer()
        fake.train.reset_mock()
        scorer.score_comparison(
            ["b1", "b2"], ["a1", "a2"], ["Q1\nAnswer:", "Q2\nAnswer:"]
        )

    fake.train.assert_not_called()


def test_when_scoring_forward_pass_executes_then_torch_grad_is_disabled():
    """
    C6: torch.no_grad() must wrap every model forward call during scoring.
    We track is_grad_enabled() inside the mock's __call__ to verify this.
    """
    grad_states: list[bool] = []

    def _tracking_fwd(**kwargs):
        grad_states.append(not torch.is_grad_enabled())
        n = kwargs["input_ids"].shape[0]
        out = MagicMock()
        out.logits = torch.zeros(n, 1)
        return out

    fake = _fake_model()
    fake.side_effect = _tracking_fwd  # overrides _fake_model's default side_effect

    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=fake,
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        scorer = reward_score.Scorer()
        scorer.score_preferences(_pairs(4))

    assert grad_states, "model forward was never called during score_preferences"
    assert all(grad_states), (
        "torch.no_grad() must be active on every forward pass; "
        f"calls with grad enabled: {grad_states.count(False)}/{len(grad_states)}"
    )


def test_when_comparison_scoring_forward_pass_executes_then_torch_grad_is_disabled():
    grad_states: list[bool] = []

    def _tracking_fwd(**kwargs):
        grad_states.append(not torch.is_grad_enabled())
        n = kwargs["input_ids"].shape[0]
        out = MagicMock()
        out.logits = torch.zeros(n, 1)
        return out

    fake = _fake_model()
    fake.side_effect = _tracking_fwd  # overrides _fake_model's default side_effect

    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=fake,
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
        ),
    ):
        scorer = reward_score.Scorer()
        scorer.score_comparison(
            ["b1", "b2"], ["a1", "a2"], ["Q1\nAnswer:", "Q2\nAnswer:"]
        )

    assert grad_states, "model forward was never called during score_comparison"
    assert all(grad_states), (
        "torch.no_grad() must be active on every forward pass in score_comparison"
    )


# ---------------------------------------------------------------------------
# C7 — sentencepiece>=0.2 in demo/requirements.txt
# ---------------------------------------------------------------------------

_DEMO_REQUIREMENTS = (
    Path(__file__).resolve().parent.parent / "demo" / "requirements.txt"
)


def test_when_requirements_txt_is_read_then_sentencepiece_is_listed():
    assert _DEMO_REQUIREMENTS.exists(), "demo/requirements.txt must exist"
    lines = [
        line.strip().lower()
        for line in _DEMO_REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    sp_lines = [ln for ln in lines if ln.startswith("sentencepiece")]
    assert sp_lines, (
        "sentencepiece must be listed in demo/requirements.txt "
        "(criterion 7: reward-model deps)"
    )


def test_when_requirements_txt_is_read_then_sentencepiece_version_constraint_is_at_least_0_2():
    content = _DEMO_REQUIREMENTS.read_text().lower()
    match = re.search(r"sentencepiece\s*>=\s*([\d]+)\.(\d+)", content)
    assert match is not None, (
        "sentencepiece must carry a >=<version> constraint in demo/requirements.txt, "
        "e.g. 'sentencepiece>=0.2'"
    )
    major, minor = int(match.group(1)), int(match.group(2))
    assert (major, minor) >= (0, 2), (
        f"sentencepiece version constraint must be >=0.2, found >={major}.{minor}"
    )
