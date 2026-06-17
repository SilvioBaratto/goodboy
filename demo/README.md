# goodboy demo — watch the weights learn manners

A tiny, runnable RLHF demo on **GPT-2 small (124M)**. It proves one point from
the explainer: *feedback changes the model's weights, and that changes its
behaviour* — not the embeddings, not a runtime filter.

We use **DPO** (Direct Preference Optimization), the modern shortcut for the
RL-optimization stage: no separate reward model, no RL loop, just preference
pairs folded into one loss. CPU-friendly.

## What happens

1. Start from raw pretrained `gpt2` — fluent but unaligned.
2. Feed it preference pairs: `chosen` = calm/constructive, `rejected` = revenge.
3. DPO nudges the weights so calm answers beat vindictive ones.
4. On **held-out** prompts, the aligned model now answers calmly — same seed,
   same decoding, only the weights differ.

```
gpt2 (base) ──DPO on preference pairs──▶ demo/aligned/
"get revenge" prompt:
   base    → vindictive ramble
   aligned → calm, constructive
```

## Run

```bash
pip install -r demo/requirements.txt

# Cycle 1 — RLAIF alignment (1000 AI-generated pairs)
python demo/train_dpo.py    # trains, saves demo/aligned/       (~hours on CPU)
python demo/compare.py      # base vs aligned, side by side

# Cycle 2 — RLHF vs RLAIF experiment
python demo/train_dpo_seed.py       # RLHF model → demo/aligned_seed/  (~5 min on CPU)
python demo/compare_rlhf_rlaif.py   # generates demo/metrics/rlhf_vs_rlaif.json
```

See `demo/RLHF_vs_RLAIF.md` for the full analysis: behavior divergence,
reward scores, weight deltas, and the bias/diversity trade-off.

## Files

| File | Role |
|------|------|
| `data/preferences.jsonl` | 1000 AI-generated preference pairs (RLAIF training) |
| `data/seed_examples.jsonl` | 12 human-written pairs (RLHF training) |
| `train_dpo.py` | DPO fine-tune `gpt2` → `aligned/` (RLAIF) |
| `train_dpo_seed.py` | DPO fine-tune `gpt2` → `aligned_seed/` (RLHF) |
| `compare.py` | before/after behaviour on held-out prompts (RLAIF) |
| `compare_rlhf_rlaif.py` | RLHF vs RLAIF comparison driver; writes `metrics/` |
| `RLHF_vs_RLAIF.md` | Analysis: divergence, diversity trade-off, honest caveats |

## Honest caveats

- GPT-2 small is a **base** model (no instruction tuning), so raw outputs
  ramble. That makes the before/after contrast clearer, but it isn't a
  production-quality assistant.
- Tiny dataset + many epochs + high LR = deliberate slight overfit so the
  behaviour shift is **visible**. Real RLHF uses far more data, ~1e-6 LR, and
  one pass. This is a teaching toy, labeled as such.
