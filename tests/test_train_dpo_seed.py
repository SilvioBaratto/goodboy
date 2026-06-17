"""Source-blind tests for demo/train_dpo_seed.py — Issue #5.

Authored from acceptance criteria only (no implementation source read).
TDD seam: module-level path constants and the pure build_config() function
allow full unit coverage without ever calling trainer.train().
"""

from pathlib import Path


# ── Module-level constants ────────────────────────────────────────────────────


def test_when_module_imported_then_seed_is_42():
    import train_dpo_seed  # type: ignore

    assert train_dpo_seed.SEED == 42


def test_when_module_imported_then_base_model_is_gpt2():
    import train_dpo_seed  # type: ignore

    assert train_dpo_seed.BASE_MODEL == "gpt2"


def test_when_module_imported_then_data_path_points_to_seed_examples_jsonl():
    import train_dpo_seed  # type: ignore

    parts = Path(train_dpo_seed.DATA).parts
    assert parts[-1] == "seed_examples.jsonl", (
        f"Expected seed_examples.jsonl, got: {parts[-1]}"
    )
    assert parts[-2] == "data", f"Expected data/, got: {parts[-2]}"
    assert parts[-3] == "demo", f"Expected demo/, got: {parts[-3]}"


def test_when_module_imported_then_output_dir_points_to_aligned_seed():
    import train_dpo_seed  # type: ignore

    parts = Path(train_dpo_seed.OUT).parts
    assert parts[-1] == "aligned_seed", f"Expected aligned_seed, got: {parts[-1]}"
    assert parts[-2] == "demo", f"Expected demo/, got: {parts[-2]}"


def test_when_module_imported_then_output_dir_is_not_bare_aligned():
    """OUT must never collide with the RLAIF model's demo/aligned/ directory."""
    import train_dpo_seed  # type: ignore

    assert Path(train_dpo_seed.OUT).name != "aligned", (
        "OUT must not point to demo/aligned/ — that directory belongs to the RLAIF model"
    )


# ── DPOConfig via build_config() ──────────────────────────────────────────────


def test_when_build_config_called_then_num_train_epochs_exceeds_rlaif_epoch_count():
    """The 12-pair RLHF run needs more epochs than the 1000-pair RLAIF run (40) for
    a deliberate visible overfit."""
    from train_dpo_seed import build_config  # type: ignore

    config = build_config()
    assert config.num_train_epochs > 40, (
        f"Expected num_train_epochs > 40 (deliberate overfit for 12 pairs), got {config.num_train_epochs}"
    )


def test_when_build_config_called_then_per_device_train_batch_size_is_2():
    from train_dpo_seed import build_config  # type: ignore

    config = build_config()
    assert config.per_device_train_batch_size == 2


def test_when_build_config_called_then_learning_rate_is_1e5():
    from train_dpo_seed import build_config  # type: ignore

    config = build_config()
    assert config.learning_rate == 1e-5


def test_when_build_config_called_then_beta_is_0_1():
    from train_dpo_seed import build_config  # type: ignore

    config = build_config()
    assert config.beta == 0.1


def test_when_build_config_called_then_max_length_is_256():
    from train_dpo_seed import build_config  # type: ignore

    config = build_config()
    assert config.max_length == 256


def test_when_build_config_called_then_max_prompt_length_is_128():
    from train_dpo_seed import build_config  # type: ignore

    config = build_config()
    assert config.max_prompt_length == 128


def test_when_build_config_called_then_precompute_ref_log_probs_is_true():
    from train_dpo_seed import build_config  # type: ignore

    config = build_config()
    assert config.precompute_ref_log_probs is True


def test_when_build_config_called_then_save_strategy_is_no():
    from train_dpo_seed import build_config  # type: ignore

    config = build_config()
    assert str(config.save_strategy) == "no"


def test_when_build_config_called_then_seed_is_42():
    from train_dpo_seed import build_config  # type: ignore

    config = build_config()
    assert config.seed == 42


def test_when_build_config_called_then_output_dir_ends_with_aligned_seed():
    from train_dpo_seed import build_config  # type: ignore

    config = build_config()
    assert Path(config.output_dir).name == "aligned_seed", (
        f"Expected output_dir to end in aligned_seed, got: {Path(config.output_dir).name}"
    )


def test_when_build_config_called_then_output_dir_does_not_end_with_bare_aligned():
    """Config output_dir must never silently point at demo/aligned/."""
    from train_dpo_seed import build_config  # type: ignore

    config = build_config()
    assert Path(config.output_dir).name != "aligned", (
        "build_config() output_dir must not be demo/aligned/ — "
        "that would overwrite the RLAIF model"
    )


# ── .gitignore ────────────────────────────────────────────────────────────────


def test_when_gitignore_read_then_demo_aligned_seed_is_listed():
    """demo/aligned_seed/ is a regeneratable build artifact and must be gitignored."""
    repo_root = Path(__file__).resolve().parent.parent
    gitignore_text = (repo_root / ".gitignore").read_text()
    assert "demo/aligned_seed" in gitignore_text, (
        "demo/aligned_seed (or demo/aligned_seed/) must appear in .gitignore"
    )
