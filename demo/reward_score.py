"""Pretrained reward-model scoring — eval only, DPO-only rule observed.

Loads OpenAssistant/reward-model-deberta-v3-base, scores:
  (a) a deterministic sample of preferences.jsonl chosen vs rejected
  (b) base vs DPO-aligned GPT-2 on held-out prompts (if demo/aligned/ exists)

SCORING / EVAL ONLY — no backward pass, no optimizer, no .train() call anywhere.
The reward model is never used for training (DPO-only rule is absolute).

One-time model download: ~270 MB (DeBERTa-v3-base, public / no-auth).
Run: python demo/reward_score.py
"""

import json
import statistics
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless CPU — must precede pyplot import
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eval.common import (
    ALIGNED,
    BASE_MODEL,
    DATA,
    HELD_OUT_PROMPTS,
    SEED,
    generate,
    load_aligned,
    load_base,
    load_tokenizer as load_gpt_tokenizer,
)
from eval.metrics_io import METRICS_DIR, write_metric

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID: str = "OpenAssistant/reward-model-deberta-v3-base"
N_SAMPLE: int = 200  # deterministic cap: first N pairs from preferences.jsonl
_CHART_NAME = "reward_scores.png"


# ── Scorer ────────────────────────────────────────────────────────────────────


class Scorer:
    """Score question/answer pairs with a pretrained reward model.

    Accepts injected model and tokenizer so unit tests can run without the
    ~270 MB checkpoint download.
    """

    def __init__(
        self,
        model: Optional[AutoModelForSequenceClassification] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
            if model is None else model
        )
        self._tokenizer = (
            AutoTokenizer.from_pretrained(MODEL_ID)
            if tokenizer is None else tokenizer
        )
        self._model.eval()  # type: ignore

    def score(self, question: str, answer: str) -> float:
        """Return a scalar reward score for a single question/answer pair."""
        enc = self._tokenizer(  # type: ignore
            question,
            text_pair=answer,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            logits = self._model(**enc).logits  # type: ignore
        return logits.squeeze().item()

    def score_preferences(self, pairs: list[dict]) -> dict:
        """Score chosen/rejected completions; return aggregate stats."""
        chosen = [self.score(p["prompt"], p["chosen"]) for p in pairs]
        rejected = [self.score(p["prompt"], p["rejected"]) for p in pairs]
        return _pref_stats(chosen, rejected)

    def score_comparison(
        self,
        base_texts: list[str],
        aligned_texts: list[str],
        prompts: list[str],
    ) -> dict:
        """Score base vs aligned generations; return per-prompt + summary stats."""
        base_s = [self.score(p, t) for p, t in zip(prompts, base_texts)]
        aligned_s = [self.score(p, t) for p, t in zip(prompts, aligned_texts)]
        return _comparison_stats(base_s, aligned_s, prompts, base_texts, aligned_texts)


# ── Pure aggregation ──────────────────────────────────────────────────────────


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    """Sample standard deviation; 0.0 for a single value (no spread)."""
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def _pref_stats(chosen: list[float], rejected: list[float]) -> dict:
    n = len(chosen)
    cm, rm = _mean(chosen), _mean(rejected)
    pct = sum(c > r for c, r in zip(chosen, rejected)) / n
    return {
        "chosen_mean": cm,
        "chosen_std": _std(chosen),
        "rejected_mean": rm,
        "rejected_std": _std(rejected),
        "margin_mean": cm - rm,
        "pct_chosen_higher": pct,
        "n_scored": n,
        "n_pairs": n,
    }


def _comparison_stats(
    base: list[float],
    aligned: list[float],
    prompts: list[str],
    base_texts: list[str],
    aligned_texts: list[str],
) -> dict:
    bm, am = _mean(base), _mean(aligned)
    # per_prompt_scores carries the full spec generations[] shape (incl. texts).
    per_prompt_scores = [
        {
            "prompt": p,
            "base_text": bt,
            "base_score": b,
            "aligned_text": at,
            "aligned_score": a,
            "delta": a - b,
        }
        for p, bt, at, b, a in zip(prompts, base_texts, aligned_texts, base, aligned)
    ]
    return {
        "base_mean": bm,
        "aligned_mean": am,
        "mean_delta": am - bm,
        "per_prompt_scores": per_prompt_scores,
    }


# ── Payload assembly ──────────────────────────────────────────────────────────


def build_payload(pref: dict, comp: dict) -> dict:
    """Assemble the full spec payload from preference + comparison stats."""
    return {
        "reward_model": MODEL_ID,
        "dataset": {
            "n_pairs": pref["n_pairs"],
            "n_scored": pref["n_scored"],
            "chosen_mean": pref["chosen_mean"],
            "chosen_std": pref["chosen_std"],
            "rejected_mean": pref["rejected_mean"],
            "rejected_std": pref["rejected_std"],
            "margin_mean": pref["margin_mean"],
            "pct_chosen_higher": pref["pct_chosen_higher"],
        },
        "generations": comp["per_prompt_scores"],
        "summary": {
            "base_mean": comp["base_mean"],
            "aligned_mean": comp["aligned_mean"],
            "mean_delta": comp["mean_delta"],
        },
        "chart_path": _CHART_NAME,
    }


# ── Persistence ───────────────────────────────────────────────────────────────


def write_metrics(payload: dict, *, output_dir: Optional[Path] = None) -> Path:
    """Render the chart and persist *payload* via the versioned metrics envelope."""
    out = Path(output_dir) if output_dir is not None else METRICS_DIR
    out.mkdir(parents=True, exist_ok=True)
    _save_chart(payload, out / _CHART_NAME)
    return write_metric(
        "reward_scores",
        payload,
        "reward_scores.json",
        base_model=BASE_MODEL,
        seed=SEED,
        output_dir=out,
    )


def _save_chart(payload: dict, path: Path) -> None:
    labels = ["base", "aligned"]
    summary = payload["summary"]
    values = [summary["base_mean"], summary["aligned_mean"]]
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Mean Reward Score")
    ax.set_title("Base vs Aligned — Mean Reward Score")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── Dataset loading ───────────────────────────────────────────────────────────


def _load_sample(n: int = N_SAMPLE) -> list[dict]:
    """Return first n rows from preferences.jsonl (deterministic)."""
    with DATA.open() as f:
        return [json.loads(line) for i, line in enumerate(f) if i < n and line.strip()]


# ── Main entry point ──────────────────────────────────────────────────────────


def _score_generations(scorer: Scorer) -> dict:
    """Generate and score base vs aligned if demo/aligned/ exists."""
    if not ALIGNED.exists():
        print("demo/aligned/ not found — skipping generation scoring. Run train_dpo.py first.")
        return {"base_mean": 0.0, "aligned_mean": 0.0, "mean_delta": 0.0, "per_prompt_scores": []}
    gpt_tok = load_gpt_tokenizer()
    base_m, aligned_m = load_base(), load_aligned()
    base_texts = [generate(base_m, gpt_tok, p) for p in HELD_OUT_PROMPTS]
    aligned_texts = [generate(aligned_m, gpt_tok, p) for p in HELD_OUT_PROMPTS]
    return scorer.score_comparison(base_texts, aligned_texts, HELD_OUT_PROMPTS)


def _print_pref(pref: dict) -> None:
    print(
        f"Dataset ({pref['n_scored']} pairs): "
        f"chosen={pref['chosen_mean']:.3f}  rejected={pref['rejected_mean']:.3f}"
        f"  margin={pref['margin_mean']:.3f}  pct_higher={pref['pct_chosen_higher']:.1%}"
    )


def _print_comp(comp: dict) -> None:
    print(
        f"Generations: base={comp['base_mean']:.3f}"
        f"  aligned={comp['aligned_mean']:.3f}  delta={comp['mean_delta']:.3f}"
    )


def main() -> None:
    """Score dataset sample and (optionally) base vs aligned generations."""
    scorer = Scorer()
    pref = scorer.score_preferences(_load_sample())
    _print_pref(pref)
    comp = _score_generations(scorer)
    _print_comp(comp)
    out = write_metrics(build_payload(pref, comp))
    print(f"Saved {out} and {out.parent / _CHART_NAME}")


if __name__ == "__main__":
    main()
