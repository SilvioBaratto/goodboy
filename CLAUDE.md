# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`goodboy` is an educational, runnable RLHF demo. The whole repo exists to prove
one claim: **feedback changes a model's weights, and that changes its behavior**
— not the embeddings, not a runtime filter. It takes raw pretrained GPT-2 small,
fine-tunes it with **DPO** on preference pairs (calm answer = `chosen`, revenge
answer = `rejected`), then shows the same prompt + same seed yields a vindictive
answer before and a calm answer after.

It is the companion code for the explainer script at
`../content-generator/previews/explained/ai_models/scripts/script_3.md`. Two
audiences at once: it drives that video AND stands alone as a clone-and-learn repo.

`README.md` is the concept explainer (the three RLHF stages). `demo/README.md`
is the run guide. `.code-generator/requirements.md` is the authoritative,
feature-by-feature spec — read it before adding scope.

## Hard scope rules (do not violate)

- **DPO only.** No PPO, no RL loop. A reward model may appear for **scoring/eval
  only**, never for training. If a task implies training against a reward signal,
  it's out of scope — confirm before building.
- The point is *behavior change*, demonstrated four ways: before/after text,
  reward/score trend (eval), per-layer weight-delta chart (blocks move,
  token-embedding matrix stays ~flat), and stage diagrams.
- Built to be **reproduced on a laptop CPU** — no GPU assumptions, fixed seeds,
  pinned deps. GPT-2 small (124M) is the model; it is a *base* model (not
  instruction-tuned), so raw outputs ramble — that's intentional contrast, keep it.
- `rejected` answers must stay mild/non-actionable (petty social nastiness only).

## Pipeline (the big picture)

```
demo/data/preferences.jsonl   ── DPO ──▶  demo/aligned/   ── compare ──▶  before vs after
   (1000 AI-generated pairs)                (fine-tuned)
demo/data/seed_examples.jsonl  (12 human-written pairs, for RLHF-vs-RLAIF contrast)
```

Three scripts, each one stage, run in order:

1. `demo/generate_dataset.py` — **RLAIF dataset generation.** Calls BAML →
   Azure OpenAI in async batches, dedups by normalized prompt, writes 1000
   `{prompt, chosen, rejected}` rows. Every prompt is formatted exactly
   `"Question: <q>\nAnswer:"`; both completions start with a leading space. This
   format is a contract — `train_dpo.py` and `compare.py` depend on it.
2. `demo/train_dpo.py` — loads `gpt2`, sets `pad_token = eos_token`, DPO
   fine-tunes on the dataset (TRL `DPOTrainer` + `DPOConfig`), saves to
   `demo/aligned/`. Epoch count is tuned for dataset size (small set → many
   epochs / deliberate overfit so the shift is visible; 1000 pairs → few epochs).
3. `demo/compare.py` — loads base `gpt2` and `demo/aligned/`, generates on
   **held-out** prompts with the same seed + greedy decoding, so weights are the
   only variable.

## Commands

```bash
pip install -r demo/requirements.txt

# Regenerate BAML client after editing any demo/baml_src/*.baml
cd demo && baml-cli generate --from baml_src

python demo/generate_dataset.py   # rebuild the 1000-pair dataset (hits Azure)
python demo/train_dpo.py          # DPO fine-tune -> demo/aligned/
python demo/compare.py            # before/after on held-out prompts

pytest                            # tests (dataset schema + pipeline smoke)
pytest tests/test_x.py::test_y    # single test
```

## BAML + Azure specifics

- Edit `demo/baml_src/*.baml` only; `demo/baml_client/` is generated — never
  hand-edit it, regenerate instead. Pinned to `baml-py` 0.222.0.
- The LLM client (`AzureFoundry`) and credentials are **borrowed from the
  sibling ITAL-IA project**: `generate_dataset.py` loads
  `/Volumes/External SSD/ITAL-IA/api/.env`. Azure OpenAI is the only provider
  configured there (deployment `gpt-4.1`, westeurope,
  api_version `2024-12-01-preview`) — there are no OpenAI/Anthropic/Google keys,
  so don't switch the client to those without adding keys.
- Dataset generation makes ~100+ real Azure calls (costs money). It is the one
  outward-facing/billable action here — confirm before re-running it.

## Regeneratable / gitignored

`demo/baml_client/` and `demo/aligned/` are build artifacts — regeneratable, keep
them out of git.
