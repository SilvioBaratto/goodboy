"""Per-layer weight-delta proof for the RLHF demo.

Loads base GPT-2 and the DPO-aligned checkpoint, computes per-layer relative
L2 norm of weight changes, persists the numbers, and renders a bar chart that
shows transformer blocks moving while the token-embedding matrix stays ~flat.

Run after training:
    python demo/train_dpo.py   # produces demo/aligned/
    python demo/weight_delta.py
"""

import math
import re
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")  # headless CPU — must precede pyplot import
import matplotlib.pyplot as plt
import torch

from eval.common import ALIGNED, BASE_MODEL, SEED, load_aligned, load_base
from eval.metrics_io import METRICS_DIR, write_metric

# ── Constants ─────────────────────────────────────────────────────────────────

_BLOCK_RE = re.compile(r"^transformer\.h\.(\d+)\.")
_CHART_NAME = "weight_delta.png"

GROUP_ORDER: list[str] = [
    "transformer.wte",
    "transformer.wpe",
    *[f"transformer.h.{i}" for i in range(12)],
    "transformer.ln_f",
]

# ── Data containers ───────────────────────────────────────────────────────────


class _ParamDelta(NamedTuple):
    name: str
    l2: float
    base_norm: float
    relative: float
    numel: int


class _GroupDelta(NamedTuple):
    group: str
    l2: float
    base_norm: float
    relative: float
    numel: int


# ── Pure math ─────────────────────────────────────────────────────────────────


def compute_param_delta(
    aligned: torch.Tensor, base: torch.Tensor
) -> tuple[float, float, float]:
    """Return (l2, base_norm, relative) for one parameter tensor."""
    with torch.no_grad():
        l2 = (aligned - base).norm().item()
        base_norm = base.norm().item()
        relative = l2 / base_norm if base_norm > 0 else 0.0
    return l2, base_norm, relative


def combine_group_l2(l2_values: list[float]) -> float:
    """Euclidean combination: sqrt(sum(l2_i^2))."""
    return math.sqrt(sum(v * v for v in l2_values))


# ── Grouping ──────────────────────────────────────────────────────────────────


def get_group_name(param_name: str) -> str:
    """Map a parameter name to its group key."""
    m = _BLOCK_RE.match(param_name)
    if m:
        return f"transformer.h.{m.group(1)}"
    return ".".join(param_name.split(".")[:2])


def validate_param_name_sets(base_names: set, aligned_names: set) -> None:
    """Raise ValueError if the two parameter name sets differ."""
    if base_names == aligned_names:
        return
    only_base = sorted(base_names - aligned_names)
    only_aligned = sorted(aligned_names - base_names)
    raise ValueError(
        f"Parameter name mismatch.\n"
        f"  Only in base:    {only_base}\n"
        f"  Only in aligned: {only_aligned}"
    )


# ── Computation ───────────────────────────────────────────────────────────────


def _compute_all_params(base_model, aligned_model) -> list[_ParamDelta]:
    """Compute per-param deltas iterating named_parameters in lockstep."""
    base_params = dict(base_model.named_parameters())
    aligned_params = dict(aligned_model.named_parameters())
    validate_param_name_sets(set(base_params), set(aligned_params))
    return [
        _ParamDelta(name, *compute_param_delta(aligned_params[name], t), t.numel())
        for name, t in base_params.items()
    ]


def _combine_group(params: list[_ParamDelta], group: str) -> _GroupDelta:
    """Euclidean-combine per-param deltas into one group entry."""
    gl2 = combine_group_l2([p.l2 for p in params])
    gbn = combine_group_l2([p.base_norm for p in params])
    return _GroupDelta(group, gl2, gbn, gl2 / gbn if gbn > 0 else 0.0,
                       sum(p.numel for p in params))


def _bucket(per_param: list[_ParamDelta]) -> dict[str, list[_ParamDelta]]:
    """Group params by transformer layer."""
    buckets: dict[str, list[_ParamDelta]] = {}
    for p in per_param:
        buckets.setdefault(get_group_name(p.name), []).append(p)
    return buckets


def _aggregate_groups(per_param: list[_ParamDelta]) -> dict[str, _GroupDelta]:
    """Return {group_name: _GroupDelta} for all parameter groups."""
    return {g: _combine_group(ps, g) for g, ps in _bucket(per_param).items()}


def build_group_deltas(base_model, aligned_model) -> dict[str, float]:
    """Return {group_name: relative_delta} for all parameter groups."""
    groups = _aggregate_groups(_compute_all_params(base_model, aligned_model))
    return {g: d.relative for g, d in groups.items()}


# ── Chart ─────────────────────────────────────────────────────────────────────


def _build_axes(group_deltas: dict[str, float]) -> tuple:  # type: ignore
    """Create figure and bar axes for the group delta chart."""
    ordered = [g for g in GROUP_ORDER if g in group_deltas]
    labels = [g.replace("transformer.", "") for g in ordered]
    fig, ax = plt.subplots(figsize=(14, 5))  # type: ignore
    ax.bar(range(len(ordered)), [group_deltas[g] for g in ordered])
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    return fig, ax


def _annotate(ax) -> None:  # type: ignore
    ax.set_ylabel("Relative L2  (‖Δθ‖ / ‖θ_base‖)")
    ax.set_title("Per-layer weight delta — DPO reshapes blocks, not embeddings")


def _write_chart(group_deltas: dict[str, float], output_dir: Path) -> None:
    fig, ax = _build_axes(group_deltas)
    _annotate(ax)
    fig.tight_layout()
    fig.savefig(output_dir / _CHART_NAME)
    plt.close(fig)


# ── Persistence ───────────────────────────────────────────────────────────────


def _build_payload(
    per_param: list[_ParamDelta], groups: dict[str, _GroupDelta]
) -> dict:
    """Assemble the full spec payload: per-param + ordered groups + chart path."""
    return {
        "per_param": [p._asdict() for p in per_param],
        "groups": [groups[g]._asdict() for g in GROUP_ORDER if g in groups],
        "chart_path": _CHART_NAME,
    }


def save_weight_delta(
    per_param: list[_ParamDelta],
    groups: dict[str, _GroupDelta],
    output_dir: Path,
) -> Path:
    """Render the chart and persist the weight-delta payload via metrics_io.

    Routes through ``metrics_io.write_metric`` so the artifact carries the
    versioned schema envelope (the issue #2 cross-cycle contract) instead of a
    bare flat dict. Returns the written JSON path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_chart({g: d.relative for g, d in groups.items()}, out)
    return write_metric(
        "weight_delta",
        _build_payload(per_param, groups),
        "weight_delta.json",
        base_model=BASE_MODEL,
        seed=SEED,
        output_dir=out,
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def _print_proof(groups: dict[str, _GroupDelta]) -> None:
    """Print the wte-vs-blocks comparison (informational, never asserts)."""
    wte = groups.get("transformer.wte")
    block_vals = [d.relative for k, d in groups.items() if k.startswith("transformer.h.")]
    if not (wte and block_vals):
        return
    mean_block = sum(block_vals) / len(block_vals)
    ratio = mean_block / wte.relative if wte.relative > 0 else float("inf")
    print(f"\nProof: wte relative={wte.relative:.6f}  mean-block relative={mean_block:.6f}")
    print(f"       Blocks moved {ratio:.1f}× more than the token embeddings.\n")


def main() -> None:
    if not ALIGNED.exists():
        raise SystemExit("No demo/aligned/ found. Run: python demo/train_dpo.py")
    print("Loading models…")
    per_param = _compute_all_params(load_base(), load_aligned())
    groups = _aggregate_groups(per_param)
    _print_proof(groups)
    save_weight_delta(per_param, groups, METRICS_DIR)
    print(f"Saved {METRICS_DIR}/weight_delta.json")
    print(f"Saved {METRICS_DIR}/{_CHART_NAME}")


if __name__ == "__main__":
    main()
