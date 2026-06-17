"""RLHF-vs-RLAIF comparison driver — consumes Cycle 1 eval tooling.

Runs base GPT-2 vs each aligned model on HELD_OUT_PROMPTS (same seed + greedy),
scores reward, computes weight-delta, and writes per-model metrics plus a
combined rlhf_vs_rlaif.json artifact.  The training dataset is the only
independent variable between the two aligned models.

Run after training both aligned models:
    python demo/train_dpo.py        # produces demo/aligned/       (RLAIF)
    python demo/train_dpo_seed.py   # produces demo/aligned_seed/  (RLHF)
    python demo/compare_rlhf_rlaif.py
"""

import json
from pathlib import Path

from eval.common import (
    ALIGNED,
    ALIGNED_SEED,
    BASE_MODEL,
    DATA,
    HELD_OUT_PROMPTS,
    SEED,
    generate,
    load_base,
    load_model_from,
    load_tokenizer,
)
from eval.metrics_io import METRICS_DIR, write_metric
from reward_score import Scorer, build_payload, write_metrics
from weight_delta import _aggregate_groups, _compute_all_params, save_weight_delta

# ── Constants ─────────────────────────────────────────────────────────────────

HERE: Path = Path(__file__).resolve().parent
DATA_RLHF: Path = HERE / "data" / "seed_examples.jsonl"

RLAIF_METRICS_DIR: Path = METRICS_DIR
RLHF_METRICS_DIR: Path = METRICS_DIR / "aligned_seed"
COMBINED_OUTPUT_FILE: str = "rlhf_vs_rlaif.json"


# ── Diversity metric ──────────────────────────────────────────────────────────


def _tokenize(text: str) -> frozenset:
    return frozenset(text.lower().strip().split())


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _diversity(pairs: list[dict]) -> dict:
    """Return prompt-diversity metrics for a preference dataset.

    distinct_prompt_ratio  — unique normalised prompts / n_pairs
    mean_pairwise_jaccard  — mean Jaccard over all prompt token-set pairs
    """
    n = len(pairs)
    if n == 0:
        return {"distinct_prompt_ratio": 0.0, "mean_pairwise_jaccard": 0.0, "n_pairs": 0}
    prompts = [p["prompt"].lower().strip() for p in pairs]
    sets = [_tokenize(p) for p in prompts]
    distinct_ratio = len(set(prompts)) / n
    pair_count = n * (n - 1) // 2
    total = sum(_jaccard(sets[i], sets[j]) for i in range(n) for j in range(i + 1, n))
    mean_jaccard = total / pair_count if pair_count > 0 else 0.0
    return {"distinct_prompt_ratio": distinct_ratio, "mean_pairwise_jaccard": mean_jaccard, "n_pairs": n}


# ── Dataset loading ───────────────────────────────────────────────────────────


def _load_pairs(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Per-model metric helpers ──────────────────────────────────────────────────


def _write_reward(scorer: Scorer, pairs: list[dict], base_texts: list[str],
                  aligned_texts: list[str], out_dir: Path) -> dict:
    """Score reward and persist metrics; return comparison stats."""
    pref = scorer.score_preferences(pairs)
    comp = scorer.score_comparison(base_texts, aligned_texts, HELD_OUT_PROMPTS)
    write_metrics(build_payload(pref, comp), output_dir=out_dir)
    return comp


def _write_delta(base_model, aligned_model, out_dir: Path) -> None:
    """Compute per-param weight delta and persist it."""
    per_param = _compute_all_params(base_model, aligned_model)
    save_weight_delta(per_param, _aggregate_groups(per_param), out_dir)


# ── Per-model orchestration ───────────────────────────────────────────────────


def _run_one_model(name: str, checkpoint: Path, dataset_path: Path, out_dir: Path,
                   scorer: Scorer, base_model, base_texts: list[str], tok) -> dict:
    """Generate, score, and persist metrics for one aligned model."""
    model = load_model_from(checkpoint)
    aligned_texts = [generate(model, tok, p) for p in HELD_OUT_PROMPTS]
    pairs = _load_pairs(dataset_path)
    comp = _write_reward(scorer, pairs, base_texts, aligned_texts, out_dir)
    _write_delta(base_model, model, out_dir)
    return {"name": name, "mean_delta": comp["mean_delta"], "aligned_texts": aligned_texts, "pairs": pairs}


# ── Combined report assembly ──────────────────────────────────────────────────


def _per_prompt_texts(rlhf_texts: list[str], rlaif_texts: list[str]) -> list[dict]:
    return [
        {"prompt": p, "rlhf": rlhf_texts[i], "rlaif": rlaif_texts[i]}
        for i, p in enumerate(HELD_OUT_PROMPTS)
    ]


def build_report(
    rlhf_reward_delta: float,
    rlaif_reward_delta: float,
    diversity_proxy,
    per_prompt_texts: list[dict],
) -> dict:
    """Assemble the pure combined-comparison dict (no I/O)."""
    return {
        "reward_deltas": {"rlhf": rlhf_reward_delta, "rlaif": rlaif_reward_delta},
        "diversity_proxy": diversity_proxy,
        "per_prompt_texts": per_prompt_texts,
    }


# ── Main entry point ──────────────────────────────────────────────────────────


def main() -> None:
    """Load base + tokenizer + Scorer ONCE; run RLAIF and RLHF; write combined."""
    tok, base, scorer = load_tokenizer(), load_base(), Scorer()
    base_texts = [generate(base, tok, p) for p in HELD_OUT_PROMPTS]
    rlaif = _run_one_model("rlaif", ALIGNED, DATA, RLAIF_METRICS_DIR, scorer, base, base_texts, tok)
    rlhf = _run_one_model("rlhf", ALIGNED_SEED, DATA_RLHF, RLHF_METRICS_DIR, scorer, base, base_texts, tok)
    report = build_report(
        rlhf_reward_delta=rlhf["mean_delta"],
        rlaif_reward_delta=rlaif["mean_delta"],
        diversity_proxy={"rlhf": _diversity(rlhf["pairs"]), "rlaif": _diversity(rlaif["pairs"])},
        per_prompt_texts=_per_prompt_texts(rlhf["aligned_texts"], rlaif["aligned_texts"]),
    )
    write_metric("rlhf_vs_rlaif", report, COMBINED_OUTPUT_FILE,
                 base_model=BASE_MODEL, seed=SEED, output_dir=RLAIF_METRICS_DIR)
    print(f"Saved {RLAIF_METRICS_DIR / COMBINED_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
