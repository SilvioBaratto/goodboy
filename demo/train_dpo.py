"""Stage 3 of RLHF (the DPO shortcut) on GPT-2 small — the 'good boy' step.

Takes the raw pretrained GPT-2, shows it pairs of (calm answer = chosen,
revenge answer = rejected), and nudges its WEIGHTS so calm answers become
more likely than vindictive ones. No reward model, no RL loop — DPO folds
the preference signal straight into a single loss.

Run:
    python demo/train_dpo.py

Output: a fine-tuned model saved to demo/aligned/. Behaviour now differs from
base gpt2 even though the architecture and tokenizer are identical — only the
weights moved. Verify with: python demo/compare.py
"""

from pathlib import Path

from datasets import load_dataset  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer  # type: ignore

SEED = 42
BASE_MODEL = "gpt2"  # 124M params, CPU-friendly, NOT instruction-tuned
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "preferences.jsonl"
OUT = HERE / "aligned"


def main() -> None:
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # GPT-2 ships without a pad token; reuse EOS so batching/DPO works.
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Explicit-prompt preference format: {prompt, chosen, rejected}.
    dataset = load_dataset("json", data_files=str(DATA), split="train")

    config = DPOConfig(
        output_dir=str(OUT),
        # Toy demo: tiny dataset + many epochs + high LR so the behaviour
        # change is clearly visible. Real runs use ~1e-6 LR and 1 epoch.
        num_train_epochs=40,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        beta=0.1,                  # KL strength: how hard we pull toward 'chosen'
        max_length=256,
        precompute_ref_log_probs=True,  # save memory on CPU
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,  # TRL v1.x name for the tokenizer
    )

    trainer.train()

    OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    print(f"\nAligned model saved to: {OUT}")
    print("Now run:  python demo/compare.py")


if __name__ == "__main__":
    main()
