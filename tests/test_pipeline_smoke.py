"""
Smoke tests for the train → compare pipeline — Issue #9.

Expected runtime: < 60 s on a modern CPU once GPT-2 is cached locally.
First-run note: Hugging Face downloads GPT-2 small on first execution
(~500 MB); allow ~2 min on a cold cache. Subsequent runs use
~/.cache/huggingface/hub/ and complete in seconds.

Run the slow suite:  pytest -m slow
Skip it:             pytest -m "not slow"
The "slow" marker is registered in pyproject.toml.

Criteria covered:
  C1 — 1-epoch DPO on 2 in-memory pairs, max_length=64, tiny batch,
        output_dir under pytest tmp_path
  C2 — Real checkpoint (config.json + *.safetensors / pytorch_model.bin)
        is produced in tmp_path
  C3 — Reloading the tmp checkpoint and calling compare.generate()
        returns a non-empty str
  C4 — demo/aligned/ is never created or clobbered when output_dir is
        tmp_path
  C5 — All slow tests skip gracefully (pytest.skip) when gpt2 is
        unreachable

Criteria skipped (NOT VERIFIABLE at runtime):
  "All tests pass"   — suite-level gate, no per-criterion assertion
  "SOLID, clean code" — subjective prose, no runtime assertion
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ALIGNED_DIR = REPO_ROOT / "demo" / "aligned"
_BASE_MODEL = "gpt2"

# Every test in this module is slow; deselect with: pytest -m "not slow"
pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Tiny in-memory dataset (format mirrors demo/data/seed_examples.jsonl:
# prompt = "Question: …\nAnswer:", completions start with one leading space)
# ---------------------------------------------------------------------------

_TINY_PAIRS = [
    {
        "prompt": "Question: A coworker took credit for my work. What should I do?\nAnswer:",
        "chosen": " Stay calm, gather the facts, and discuss it directly.",
        "rejected": " Start spreading rumors about them to get back.",
    },
    {
        "prompt": "Question: Someone spread lies about me. How do I respond?\nAnswer:",
        "chosen": " Correct the record calmly with the people who matter.",
        "rejected": " Spread worse lies about them first.",
    },
]


# ---------------------------------------------------------------------------
# Helpers — each < 10 lines, single responsibility
# ---------------------------------------------------------------------------


def _load_gpt2():
    """Return (model, tokenizer); raises OSError / ConnectionError when offline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(_BASE_MODEL)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def _dpo_config_kwargs(output_dir: Path) -> dict:
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "max_length": 64,
        "precompute_ref_log_probs": False,  # True + Dataset.from_list breaks cache on in-memory data
        "save_strategy": "no",
        "report_to": "none",
        "seed": 42,
    }


def _build_dpo_config(output_dir: Path):
    from trl.trainer.dpo_config import DPOConfig  # type: ignore

    config = DPOConfig(**_dpo_config_kwargs(output_dir))
    config.max_prompt_length = (  # type: ignore
        32  # removed from ctor in TRL 1.x; see train_dpo_seed.py:55
    )
    config.save_strategy = "no"  # reset enum → plain string; see train_dpo_seed.py:56
    return config


def _build_trainer(model, tokenizer, config):
    from datasets import Dataset
    from trl.trainer.dpo_trainer import DPOTrainer  # type: ignore

    return DPOTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(_TINY_PAIRS),
        processing_class=tokenizer,
    )


def _run_training(model, tokenizer, output_dir: Path) -> None:
    """1-epoch DPO on _TINY_PAIRS; persist checkpoint to output_dir."""
    trainer = _build_trainer(model, tokenizer, _build_dpo_config(output_dir))
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))  # save_strategy="no" → must call explicitly
    tokenizer.save_pretrained(str(output_dir))


# ---------------------------------------------------------------------------
# Module-scoped fixture — trains once, shared by C1/C2/C3.
# pytest.skip() inside a fixture propagates the skip to every dependent test.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory):
    """Load gpt2, run 1-epoch DPO, return (checkpoint_dir, tokenizer)."""
    try:
        model, tokenizer = _load_gpt2()
    except OSError as exc:
        pytest.skip(f"gpt2 unavailable (offline or not cached): {exc}")
    out_dir = tmp_path_factory.mktemp("aligned_smoke")
    _run_training(model, tokenizer, out_dir)
    return out_dir, tokenizer


# ---------------------------------------------------------------------------
# C1 — 1-epoch DPO run completes without raising
# ---------------------------------------------------------------------------


def test_when_dpo_trained_on_tiny_pairs_then_run_completes(trained_checkpoint):
    """C1: _run_training must not raise; output_dir must exist afterward."""
    out_dir, _ = trained_checkpoint
    assert out_dir.is_dir()


# ---------------------------------------------------------------------------
# C2a — config.json is produced in the checkpoint directory
# ---------------------------------------------------------------------------


def test_when_dpo_trained_then_config_json_is_produced(trained_checkpoint):
    """C2a: output_dir must contain config.json after training."""
    out_dir, _ = trained_checkpoint
    assert (out_dir / "config.json").exists(), "config.json missing from output_dir"


# ---------------------------------------------------------------------------
# C2b — at least one model-weights file is produced
# ---------------------------------------------------------------------------


def test_when_dpo_trained_then_weights_file_is_produced(trained_checkpoint):
    """C2b: output_dir must contain *.safetensors or pytorch_model.bin."""
    out_dir, _ = trained_checkpoint
    has_safetensors = bool(list(out_dir.glob("*.safetensors")))
    has_bin = (out_dir / "pytorch_model.bin").exists()
    assert has_safetensors or has_bin, "no weights file found in output_dir"


# ---------------------------------------------------------------------------
# C3 — Reloading checkpoint + compare.generate() returns non-empty str
# ---------------------------------------------------------------------------


def test_when_checkpoint_reloaded_then_generate_returns_nonempty_string(
    trained_checkpoint,
):
    """C3: compare.generate(model, tokenizer, prompt) must return a non-empty str."""
    from transformers import AutoModelForCausalLM

    import compare

    out_dir, tokenizer = trained_checkpoint
    model = AutoModelForCausalLM.from_pretrained(str(out_dir)).eval()
    result = compare.generate(model, tokenizer, compare.PROMPTS[0])

    assert isinstance(result, str), (
        f"generate() returned {type(result).__name__}, not str"
    )
    assert result.strip(), "generate() returned an empty string"


# ---------------------------------------------------------------------------
# C4 — demo/aligned/ is never created/clobbered when output_dir is tmp_path
#
# Trains independently (not via the module fixture) so the pre-training state
# of ALIGNED_DIR is captured before any training run in this test.
# ---------------------------------------------------------------------------


def test_when_dpo_trained_with_tmp_output_then_aligned_dir_is_untouched(tmp_path):
    """C4: demo/aligned/ must not be created or modified when output_dir is tmp_path."""
    try:
        model, tokenizer = _load_gpt2()
    except OSError as exc:
        pytest.skip(f"gpt2 unavailable: {exc}")

    existed_before = ALIGNED_DIR.exists()
    mtime_before = ALIGNED_DIR.stat().st_mtime if existed_before else None

    _run_training(model, tokenizer, tmp_path / "aligned_smoke")

    if not existed_before:
        assert not ALIGNED_DIR.exists(), (
            "demo/aligned/ must not be created by the smoke run"
        )
    else:
        assert ALIGNED_DIR.stat().st_mtime == mtime_before, (
            "demo/aligned/ must not be modified by the smoke run"
        )
