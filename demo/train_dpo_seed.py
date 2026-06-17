"""DPO fine-tune GPT-2 on the 12 human-written seed pairs (RLHF arm).

This is the human-feedback half of the RLHF-vs-RLAIF experiment. It produces a
second aligned checkpoint in demo/aligned_seed/ that is directly comparable to
the AI-trained model in demo/aligned/ — same base model, same decoding, same
held-out prompts; only the training dataset differs.

Honest framing: 12 pairs × 60 epochs is a deliberate teaching overfit. On a
dataset this small, many epochs are needed to make the behaviour shift visible
on held-out prompts. This mirrors the toy-overfit framing in demo/README.md;
do not interpret the high epoch count as a production best practice.

Run:
    python demo/train_dpo_seed.py

Output: a fine-tuned model saved to demo/aligned_seed/. Compare with:
    python demo/compare_rlhf_rlaif.py
"""

from pathlib import Path

from datasets import load_dataset  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer  # type: ignore

SEED: int = 42
BASE_MODEL: str = "gpt2"
HERE: Path = Path(__file__).resolve().parent
DATA: Path = HERE / "data" / "seed_examples.jsonl"
OUT: Path = HERE / "aligned_seed"


def build_config() -> DPOConfig:
    """Return DPOConfig for the 12-pair seed run.

    Mirrors train_dpo.py exactly EXCEPT num_train_epochs, which is raised to 60
    for a deliberately visible overfit on the tiny 12-pair human dataset.
    max_prompt_length was removed from DPOConfig in TRL 1.x; preserved as an
    attribute for API parity with train_dpo.py and downstream consumers.
    """
    config = DPOConfig(
        output_dir=str(OUT),
        num_train_epochs=60,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        beta=0.1,
        max_length=256,
        precompute_ref_log_probs=True,
        bf16=False,  # TRL defaults bf16=True; force fp32 for CPUs without bf16
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )
    config.max_prompt_length = 128  # type: ignore[attr-defined]
    config.save_strategy = "no"  # type: ignore[assignment]  # reset enum → plain string
    return config


def main() -> None:
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("json", data_files=str(DATA), split="train")

    trainer = DPOTrainer(
        model=model,
        args=build_config(),
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    print(f"\nSeed-aligned model saved to: {OUT}")
    print("Now run:  python demo/compare_rlhf_rlaif.py")


if __name__ == "__main__":
    main()
