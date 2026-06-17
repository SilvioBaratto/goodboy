"""
Source-blind tests for issue #10:
  docs: stage diagrams (SFT -> RM -> DPO; PPO alternative) and README results section

Oracle-verifiable criteria tested here:

  [UNIT] README gains a ## Results section with: real before/after text (from
         rlhf_vs_rlaif.json / RLHF_vs_RLAIF.md §3), the reward-score trend (§4),
         and the weight-delta chart/signature (§5), plus a link to demo/RLHF_vs_RLAIF.md

  [UNIT] All cited numbers are clearly labeled as illustrative samples; toy-overfit /
         base-model caveats retained; any chart image reference notes it is a gitignored
         generated artifact (regenerate via `python demo/weight_delta.py`)

  [UNIT] The three guard tests in tests/test_rlhf_vs_rlaif_doc.py are flipped to
         positive assertions (README *does* have the Results heading / diagram), not deleted

Skipped (oracle: NOT VERIFIABLE):
  - README gains stage diagrams for SFT -> Reward Model -> DPO with PPO annotated as
    alternative path (diagram content/accuracy, not runtime-verifiable)
  - Roadmap: check off only genuinely-delivered items (content accuracy, not verifiable)
  - All tests pass (boilerplate suite gate)
  - SOLID, clean code, TDD (subjective prose)
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
README_PATH = REPO_ROOT / "README.md"
GUARD_TEST_PATH = Path(__file__).parent / "test_rlhf_vs_rlaif_doc.py"


def _results_section(text: str) -> str:
    """Extract the content of the first Results-titled heading section (any heading level)."""
    m = re.search(r"^(#{1,6})\s+Results\b[^\n]*\n", text, re.MULTILINE | re.IGNORECASE)
    if not m:
        return ""
    level = len(m.group(1))
    after_heading = text[m.end() :]
    nxt = re.search(r"^#{1," + str(level) + r"}\s", after_heading, re.MULTILINE)
    body = after_heading[: nxt.start()] if nxt else after_heading
    return m.group(0) + body


# ---------------------------------------------------------------------------
# Criterion: README gains a ## Results section with specific content
# ---------------------------------------------------------------------------


def test_when_readme_is_read_then_results_section_heading_is_present():
    """README.md must contain a heading named 'Results' at any markdown level."""
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    assert re.search(r"^#{1,6}\s+Results\b", text, re.MULTILINE | re.IGNORECASE), (
        "README.md must contain a '## Results' (or equivalently-levelled) section heading"
    )


def test_when_readme_is_read_then_results_section_links_to_rlhf_vs_rlaif_doc():
    """The Results section must link to demo/RLHF_vs_RLAIF.md (the full analysis document)."""
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    section = _results_section(text)
    assert section, "README.md must have a Results section"
    assert "demo/RLHF_vs_RLAIF.md" in section, (
        "The Results section must contain a link to demo/RLHF_vs_RLAIF.md"
    )


def test_when_readme_is_read_then_results_section_contains_before_and_after_text():
    """
    The Results section must show real before/after model output (§3 of RLHF_vs_RLAIF.md).
    Simplest consistent interpretation: both 'before' and 'after' appear in the section,
    reflecting the behavioral comparison between the base and aligned models.
    """
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    section = _results_section(text).lower()
    assert section, "README.md must have a Results section"
    assert "before" in section, (
        "Results section must reference pre-DPO ('before') model output"
    )
    assert "after" in section, (
        "Results section must reference post-DPO ('after') model output"
    )


def test_when_readme_is_read_then_results_section_references_reward_score_trend():
    """
    The Results section must reference the reward-score trend (§4 of RLHF_vs_RLAIF.md).
    'reward' or 'score' must appear within the section.
    """
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    section = _results_section(text).lower()
    assert section, "README.md must have a Results section"
    assert "reward" in section or "score" in section, (
        "Results section must reference the reward-score trend (§4)"
    )


def test_when_readme_is_read_then_results_section_references_weight_delta():
    """
    The Results section must reference the weight-delta chart/signature (§5 of RLHF_vs_RLAIF.md).
    Both 'weight' and 'delta' must appear in the section.
    """
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    section = _results_section(text).lower()
    assert section, "README.md must have a Results section"
    assert "weight" in section, (
        "Results section must reference the weight-delta analysis (§5)"
    )
    assert "delta" in section, (
        "Results section must reference the weight-delta analysis (§5)"
    )


# ---------------------------------------------------------------------------
# Criterion: illustrative labels, caveats, and chart-artifact annotation
# ---------------------------------------------------------------------------


def test_when_readme_is_read_then_numbers_are_labeled_illustrative():
    """Cited numbers must be labeled as 'illustrative' samples, not authoritative metrics."""
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    assert "illustrative" in text.lower(), (
        "README.md must label cited numbers as 'illustrative' samples"
    )


def test_when_readme_is_read_then_toy_overfit_caveat_is_present():
    """
    The toy-overfit caveat must be retained.
    'toy' or 'overfit' must appear somewhere in README.md.
    """
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    lower = text.lower()
    assert "toy" in lower or "overfit" in lower, (
        "README.md must retain the toy-overfit caveat"
    )


def test_when_readme_is_read_then_base_model_caveat_is_present():
    """
    The base-model caveat must be retained (GPT-2 is a base model; raw outputs will ramble).
    'base' model terminology must appear in README.md.
    """
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    assert "base" in text.lower(), "README.md must retain the base-model caveat"


def test_when_readme_is_read_then_chart_image_references_weight_delta_py():
    """
    Any chart image reference must note regeneration via 'python demo/weight_delta.py'.
    The filename 'weight_delta.py' must appear in README.md.
    """
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    assert "weight_delta.py" in text, (
        "README.md must note that the weight-delta chart is regenerated via "
        "'python demo/weight_delta.py'"
    )


def test_when_readme_is_read_then_chart_is_noted_as_gitignored_generated_artifact():
    """
    Chart image references must note the image is a gitignored, regeneratable artifact.
    'gitignore', 'generated', or 'regenerate' must appear in README.md.
    """
    assert README_PATH.exists(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    lower = text.lower()
    assert "gitignore" in lower or "generated" in lower or "regenerate" in lower, (
        "README.md must note that chart images are gitignored generated artifacts"
    )


# ---------------------------------------------------------------------------
# Criterion: three guard tests in test_rlhf_vs_rlaif_doc.py flipped, not deleted
# ---------------------------------------------------------------------------


def test_when_guard_test_file_existence_is_checked_then_file_has_not_been_deleted():
    """
    Guard tests must be FLIPPED to positive assertions, not deleted.
    tests/test_rlhf_vs_rlaif_doc.py must still exist.
    """
    assert GUARD_TEST_PATH.exists(), (
        "tests/test_rlhf_vs_rlaif_doc.py must not be deleted; "
        "its three guard tests must be flipped to positive assertions instead"
    )


def test_when_guard_test_file_is_read_then_at_least_three_tests_are_still_present():
    """
    Flipping means replacing with positive counterparts, not removing.
    The file must still contain at least three test functions.
    """
    assert GUARD_TEST_PATH.exists(), f"{GUARD_TEST_PATH} must exist"
    text = GUARD_TEST_PATH.read_text(encoding="utf-8")
    tests = re.findall(r"^def test_", text, re.MULTILINE)
    assert len(tests) >= 3, (
        "tests/test_rlhf_vs_rlaif_doc.py must retain at least 3 test functions "
        "(flip guards to positive assertions — do not delete them)"
    )


def test_when_guard_test_file_is_read_then_negative_results_heading_guard_is_absent():
    """
    The negative guard 'no_results_section_heading_is_present' must be renamed/removed.
    This function name encodes the old negative assertion that must be flipped to positive.
    """
    assert GUARD_TEST_PATH.exists(), f"{GUARD_TEST_PATH} must exist"
    text = GUARD_TEST_PATH.read_text(encoding="utf-8")
    assert "no_results_section_heading_is_present" not in text, (
        "tests/test_rlhf_vs_rlaif_doc.py must no longer contain "
        "'no_results_section_heading_is_present'; replace it with a positive assertion"
    )


def test_when_guard_test_file_is_read_then_negative_embedded_image_guard_is_absent():
    """
    The negative guard 'no_embedded_image_markup_is_present' must be renamed/removed.
    """
    assert GUARD_TEST_PATH.exists(), f"{GUARD_TEST_PATH} must exist"
    text = GUARD_TEST_PATH.read_text(encoding="utf-8")
    assert "no_embedded_image_markup_is_present" not in text, (
        "tests/test_rlhf_vs_rlaif_doc.py must no longer contain "
        "'no_embedded_image_markup_is_present'; replace it with a positive assertion"
    )


def test_when_guard_test_file_is_read_then_negative_mermaid_guard_is_absent():
    """
    The negative guard 'no_mermaid_diagram_block_is_present' must be renamed/removed.
    """
    assert GUARD_TEST_PATH.exists(), f"{GUARD_TEST_PATH} must exist"
    text = GUARD_TEST_PATH.read_text(encoding="utf-8")
    assert "no_mermaid_diagram_block_is_present" not in text, (
        "tests/test_rlhf_vs_rlaif_doc.py must no longer contain "
        "'no_mermaid_diagram_block_is_present'; replace it with a positive assertion"
    )
