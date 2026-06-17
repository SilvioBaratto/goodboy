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

*Educational repo. No training code yet — see the roadmap below.*

## Roadmap

- [ ] Minimal reward-model training notebook
- [ ] Toy PPO loop on a small model
- [ ] DPO comparison example
- [ ] Diagrams for each stage
