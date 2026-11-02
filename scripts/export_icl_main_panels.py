#!/usr/bin/env python3
"""Export the curves behind Fig. 4 as one NPZ per panel.

Writes:
  - panel_A_training_curve.npz
  - panel_B_block2_attention.npz
  - panel_C_block2_mlp.npz
  - shared_baselines.npz
under <results-dir>/<output-subdir>.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def parse_alpha_from_name(path: Path) -> float:
    m = re.search(r"_a([0-9]+p[0-9]+|[0-9]+)_mean\.npz$", path.name)
    if not m:
        raise ValueError(f"Could not parse alpha suffix from: {path.name}")
    return float(m.group(1).replace("p", "."))


def nearest_available(target: float, available: list[float]) -> float:
    return min(available, key=lambda x: abs(x - target))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", type=str, default="results/icl")
    ap.add_argument("--means-subdir", type=str, default="fisher_component_wse_alpha_means")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--output-subdir", type=str, default="block1_panel_npz")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    means_dir = (results_dir / args.means_subdir).resolve()
    out_dir = (results_dir / args.output_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_files = sorted(means_dir.glob("fisher_component_projection_wse_a*_mean.npz"))
    if not mean_files:
        raise FileNotFoundError(f"No per-alpha mean files found under {means_dir}")

    by_alpha = {}
    for fp in mean_files:
        d = np.load(fp, allow_pickle=True)
        alpha = (
            float(d["_ensemble_alpha"])
            if "_ensemble_alpha" in d.files
            else parse_alpha_from_name(fp)
        )
        by_alpha[alpha] = d
    chosen_alpha = nearest_available(args.alpha, sorted(by_alpha.keys()))
    d = by_alpha[chosen_alpha]

    if "block1_attn_test_mean" not in d.files or "block1_mlp_test_mean" not in d.files:
        raise KeyError("Missing block1_attn/block1_mlp curves in selected alpha mean file.")

    # Mean best held-out loss achieved during training, the Fig. 4 baseline.
    best_held_per_seed = []
    ex_seed = None
    ex_curve = None
    seed_dirs = sorted(
        (p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    for sd in seed_dirs:
        tc = sd / "training_curves.npz"
        if not tc.is_file():
            continue
        t = np.load(tc)
        if "held_out_loss_history" in t.files:
            h = np.asarray(t["held_out_loss_history"], dtype=np.float64)
            if h.size:
                best_held_per_seed.append(float(np.min(h)))
        if ex_curve is None and all(
            k in t.files for k in ("log_steps", "train_task_loss_history", "held_out_loss_history")
        ):
            ex_seed = int(sd.name.split("_")[-1])
            ex_curve = (
                np.asarray(t["log_steps"], dtype=np.float64),
                np.asarray(t["train_task_loss_history"], dtype=np.float64),
                np.asarray(t["held_out_loss_history"], dtype=np.float64),
            )
        t.close()

    if ex_curve is None:
        raise FileNotFoundError("No seed_*/training_curves.npz with required fields for panel A.")

    ref_test = float(d["_ref_test_mean"]) if "_ref_test_mean" in d.files else np.nan
    ref_train = float(d["_ref_train_mean"]) if "_ref_train_mean" in d.files else np.nan
    best_held_ref = float(np.mean(best_held_per_seed)) if best_held_per_seed else np.nan
    n_seed_here = int(d["_n_seeds"]) if "_n_seeds" in d.files else -1

    np.savez(
        out_dir / "shared_baselines.npz",
        ensemble_alpha=np.float64(chosen_alpha),
        ref_test_mean=np.float64(ref_test),
        ref_train_mean=np.float64(ref_train),
        early_stopping_test_mean=np.float64(best_held_ref),
        n_seeds_curve_aggregate=np.int64(n_seed_here),
        n_seed_dirs=np.int64(len(seed_dirs)),
        n_seeds_for_early_stopping=np.int64(len(best_held_per_seed)),
        block_slug=np.asarray("block1"),
        block_display=np.asarray("Block 2"),
    )

    steps, train_hist, test_hist = ex_curve
    best_idx = int(np.argmin(test_hist))
    np.savez(
        out_dir / "panel_A_training_curve.npz",
        panel_label=np.asarray("A"),
        example_seed=np.int64(ex_seed if ex_seed is not None else -1),
        log_steps=steps,
        train_task_loss_history=train_hist,
        held_out_loss_history=test_hist,
        best_step=np.float64(steps[best_idx]),
        best_test=np.float64(test_hist[best_idx]),
        final_step=np.float64(steps[-1]),
        final_test=np.float64(test_hist[-1]),
        ensemble_alpha=np.float64(chosen_alpha),
    )

    for panel, slug, component in [
        ("B", "block1_attn", "attention"),
        ("C", "block1_mlp", "mlp"),
    ]:
        ks = np.asarray(d[f"{slug}_ks"])
        train_mean = np.asarray(d[f"{slug}_train_mean"])
        test_mean = np.asarray(d[f"{slug}_test_mean"])
        train_std = (
            np.asarray(d[f"{slug}_train_std"])
            if f"{slug}_train_std" in d.files
            else np.zeros_like(train_mean)
        )
        test_std = (
            np.asarray(d[f"{slug}_test_std"])
            if f"{slug}_test_std" in d.files
            else np.zeros_like(test_mean)
        )
        np.savez(
            out_dir / f"panel_{panel}_block2_{component}.npz",
            panel_label=np.asarray(panel),
            group_slug=np.asarray(slug),
            display_label=np.asarray(f"Block 2 {component.title()}"),
            block_index_zero=np.int64(1),
            block_index_one=np.int64(2),
            component=np.asarray(component),
            ensemble_alpha=np.float64(chosen_alpha),
            ks=ks,
            train_mean=train_mean,
            train_std=train_std,
            test_mean=test_mean,
            test_std=test_std,
            ref_test_mean=np.float64(ref_test),
            ref_train_mean=np.float64(ref_train),
            early_stopping_test_mean=np.float64(best_held_ref),
            n_seeds_curve_aggregate=np.int64(n_seed_here),
        )

    print(f"Wrote block1 panel NPZs to: {out_dir}")
    print(f"Used alpha: {chosen_alpha} (requested {args.alpha})")


if __name__ == "__main__":
    main()
