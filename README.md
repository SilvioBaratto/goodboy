# 🐕 goodboy

> How **Reinforcement Learning from Human Feedback (RLHF)** turns a raw language model into a helpful, aligned assistant — explained from scratch.

RLHF is the training recipe behind models like ChatGPT, Claude, and Gemini. The name of this repo is the intuition: you train a model the way you train a dog — not by writing rules, but by rewarding good behavior. Say *"good boy"* enough times for the right thing, and the behavior sticks.

---

## TL;DR

A pretrained language model knows *how language works* but not *how you want it to behave*. RLHF fixes the second part in three stages:

1. **SFT** — Show it examples of good answers (supervised fine-tuning).
2. **Reward Model** — Teach a separate model to score answers the way humans would.
3. **RL optimization** — Let the language model practice, using the reward model as the "good boy / bad boy" signal.

```
Pretrained LLM ──▶ [1] SFT ──▶ [2] Reward Model ──▶ [3] PPO / DPO ──▶ Aligned model
```

---

## Why RLHF exists

Pretraining optimizes one thing: **predict the next token** over a giant pile of internet text. That gives a model fluent, knowledgeable — but it has no idea what *you* consider a good response. Ask it a question and it might:

- continue your question instead of answering it,
- produce something plausible but unhelpful,
- happily generate harmful or false content.

You *could* try to write down rules for "good behavior," but human preferences are fuzzy, contextual, and hard to specify. The RLHF insight: **don't specify the reward — learn it from human comparisons.**

---

## The three stages

### Stage 1 — Supervised Fine-Tuning (SFT)

Collect a dataset of high-quality `(prompt, ideal answer)` pairs written by humans, then fine-tune the pretrained model on them with standard next-token prediction.

This gets the model into the right "format" — answering questions, following instructions — but it only knows the demonstrations it was shown. It can't generalize preferences it never saw.

> **Output:** a model that *responds* reasonably. The starting point for the next stages.

### Stage 2 — Reward Model (RM)

Humans are bad at writing the perfect answer, but great at **comparing** two answers. So:

1. Sample several answers from the SFT model for the same prompt.
2. A human ranks them: *A is better than B*.
3. Train a **reward model** to predict a scalar score such that preferred answers score higher.

The standard loss is the **Bradley–Terry** pairwise objective:

```
loss = −log( σ( r(prompt, chosen) − r(prompt, rejected) ) )
```

where `r(·)` is the reward model's score and `σ` is the sigmoid. Intuitively: push the chosen answer's score above the rejected one's.

> **Output:** a model that, given any answer, outputs a number ≈ "how much a human would like this."

### Stage 3 — RL Optimization (PPO or DPO)

Now optimize the SFT model to produce answers the reward model scores highly.

**PPO (Proximal Policy Optimization)** — the classic approach:

- The language model is the **policy**; each generated answer gets a reward from the RM.
- A **KL-divergence penalty** keeps the policy from drifting too far from the SFT model — otherwise it learns to "hack" the reward model with gibberish that scores high but reads badly.

```
objective = E[ r(prompt, answer) ] − β · KL( policy ‖ SFT_reference )
                  ▲ maximize reward        ▲ but stay close to the original
```

**DPO (Direct Preference Optimization)** — the modern shortcut:

- Skips training a separate reward model *and* the RL loop.
- Optimizes directly on the preference pairs with a clever closed-form loss that's mathematically equivalent to the RLHF objective.
- Simpler, more stable, cheaper — now the default for many open models.

> **Output:** the aligned model. Helpful, follows instructions, refuses harmful requests — a good boy. 🐕

---

## Reward hacking (the thing everyone fights)

The reward model is a *proxy* for human preference, not the real thing. Optimize hard enough against any proxy and the policy finds exploits — verbose padding, sycophancy, confident nonsense that scores high. This is why:

- the **KL penalty** exists (don't stray far from sane language),
- reward models get **retrained** as the policy discovers new exploits,
- **human eval** stays in the loop — the proxy is never fully trusted.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Policy** | The language model being trained; it "acts" by generating tokens. |
| **Reward model (RM)** | Learned scorer that approximates human preference. |
| **KL penalty** | Term keeping the policy close to the reference model. |
| **PPO** | RL algorithm that updates the policy from rewards, conservatively. |
| **DPO** | Trains directly on preference pairs, no separate RM or RL loop. |
| **Reward hacking** | Policy exploiting flaws in the reward model instead of being genuinely good. |

---

## Further reading

- Christiano et al., 2017 — *Deep RL from Human Preferences* (the original idea)
- Ouyang et al., 2022 — *InstructGPT* (RLHF for instruction following)
- Rafailov et al., 2023 — *Direct Preference Optimization (DPO)*
- Hugging Face **TRL** library — practical RLHF/DPO implementations

---

## Stage Diagrams

### Three-stage RLHF pipeline

```
Pretrained LLM
    │
    ▼
[1] SFT ── supervised fine-tuning on high-quality (prompt, answer) pairs
    │
    ▼
[2] Reward Model ── trained on human preference comparisons (A > B)
    │           scores any answer: how much would a human like this?
    │
    ├──▶ [3a] PPO  (classic RL path — InstructGPT / GPT-3.5)
    │          objective = E[r(prompt, answer)] − β · KL(policy ‖ SFT_reference)
    │          KL-penalty β keeps the policy from reward-hacking
    │
    └──▶ [3b] DPO  (direct preference optimisation — this repo)
               skips the RL loop; solves the same objective in closed form
               using only the preference pairs — simpler, more stable
```

DPO and PPO are alternative Stage 3 optimisers that share the same RLHF goal.
PPO requires a running reward model and a KL-penalised RL loop; DPO collapses
both into a single supervised loss. **This repo uses DPO only** — PPO is shown
above as the alternative path for conceptual completeness.

---

## Results

> **Numbers below are illustrative samples** from a toy training run on GPT-2
> small (124 M parameters, base model — not instruction-tuned). The deliberately
> high epoch count causes the model to overfit its training distribution so the
> behaviour shift is large and clearly visible on a laptop CPU; a real production
> run would use far more data and far fewer epochs.
> See [demo/RLHF_vs_RLAIF.md](demo/RLHF_vs_RLAIF.md) for the full analysis.

### Before and after alignment (§3)

**Prompt:** "A stranger was rude to me online. How do I get revenge?"

**Before DPO** (raw base GPT-2): the model rambles or trails off into incoherent
text — expected for a base model that has never been shaped toward Q&A.

**After DPO** (aligned): *"Honestly, the best response is usually no response.
Block them, move on, and don't let a stranger's bad behaviour ruin your day.
Your time and energy are better spent elsewhere."*

Both aligned models redirect the revenge framing toward constructive alternatives.
Full before/after texts for all three held-out prompts are in
`demo/metrics/rlhf_vs_rlaif.json` and §3 of [demo/RLHF_vs_RLAIF.md](demo/RLHF_vs_RLAIF.md).

### Reward score trend (§4)

A pretrained reward model (`OpenAssistant/reward-model-deberta-v3-base`) scores
the held-out generations. Mean reward score delta over the unaligned base
(illustrative samples from `demo/metrics/rlhf_vs_rlaif.json`):

| Model | Mean reward delta (aligned − base) |
|---|---|
| RLHF (12 human pairs)  | **+0.23** |
| RLAIF (1000 AI pairs)  | **+0.31** |

Both aligned models improve over the unaligned base. Full per-prompt breakdown:
`demo/metrics/reward_scores.json`.

### Weight delta (§5)

DPO shifts the transformer's attention and MLP weights without meaningfully
changing the token-embedding matrix (`transformer.wte`) — proof that RLHF
reshapes *reasoning*, not *vocabulary*.

![Per-layer weight delta chart](demo/metrics/weight_delta.png)

> _The chart above is a **gitignored generated artifact** — not committed to the
> repo. Regenerate it after training:_
>
> ```bash
> python demo/weight_delta.py
> ```

Per-layer weight-delta data: `demo/metrics/weight_delta.json`.

---

## Roadmap

- [ ] Minimal reward-model training notebook
- [ ] Toy PPO loop on a small model
- [x] DPO comparison example
- [x] Diagrams for each stage
