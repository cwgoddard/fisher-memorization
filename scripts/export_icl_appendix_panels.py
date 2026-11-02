#!/usr/bin/env python3
"""Export the curves behind Fig. 8 as one NPZ per panel.

Writes, under <results-dir>/<output-subdir>:
  - shared_baselines.npz
  - panel_A_block1_attention.npz ... panel_F_block3_mlp.npz

The payloads are taken directly from the alpha-mean NPZ.
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
    ap.add_argument("--output-subdir", type=str, default="appendix_panel_npz")
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

    expected = [
        "block0_attn",
        "block0_mlp",
        "block1_attn",
        "block1_mlp",
        "block2_attn",
        "block2_mlp",
    ]
    slugs = [s for s in expected if f"{s}_test_mean" in d.files]
    if not slugs:
        raise ValueError("No expected block component curves found in alpha mean file.")

    label_map = {
        "block0_attn": "Block 1 Attention",
        "block0_mlp": "Block 1 MLP",
        "block1_attn": "Block 2 Attention",
        "block1_mlp": "Block 2 MLP",
        "block2_attn": "Block 3 Attention",
        "block2_mlp": "Block 3 MLP",
    }
    panel_labels = ["A", "B", "C", "D", "E", "F"]

    ref_test = float(d["_ref_test_mean"]) if "_ref_test_mean" in d.files else np.nan
    ref_train = float(d["_ref_train_mean"]) if "_ref_train_mean" in d.files else np.nan

    best_held_per_seed = []
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
        t.close()
    best_held_ref = float(np.mean(best_held_per_seed)) if best_held_per_seed else np.nan

    np.savez(
        out_dir / "shared_baselines.npz",
        ensemble_alpha=np.float64(chosen_alpha),
        ref_test_mean=np.float64(ref_test),
        ref_train_mean=np.float64(ref_train),
        early_stopping_test_mean=np.float64(best_held_ref),
        n_seed_dirs=np.int64(len(seed_dirs)),
        n_seeds_for_early_stopping=np.int64(len(best_held_per_seed)),
        panel_slugs=np.asarray(slugs, dtype=object),
    )

    for i, slug in enumerate(slugs):
        ks = np.asarray(d[f"{slug}_ks"])
        test_mean = np.asarray(d[f"{slug}_test_mean"])
        test_std = (
            np.asarray(d[f"{slug}_test_std"])
            if f"{slug}_test_std" in d.files
            else np.zeros_like(test_mean)
        )
        train_mean = (
            np.asarray(d[f"{slug}_train_mean"])
            if f"{slug}_train_mean" in d.files
            else np.full_like(test_mean, np.nan)
        )
        train_std = (
            np.asarray(d[f"{slug}_train_std"])
            if f"{slug}_train_std" in d.files
            else np.zeros_like(test_mean)
        )

        block_index_zero = int(slug[5])
        component = "attention" if slug.endswith("_attn") else "mlp"
        panel = panel_labels[i]
        out_name = f"panel_{panel}_block{block_index_zero + 1}_{component}.npz"

        np.savez(
            out_dir / out_name,
            panel_label=np.asarray(panel),
            group_slug=np.asarray(slug),
            display_label=np.asarray(label_map.get(slug, slug)),
            block_index_zero=np.int64(block_index_zero),
            block_index_one=np.int64(block_index_zero + 1),
            component=np.asarray(component),
            ensemble_alpha=np.float64(chosen_alpha),
            ks=ks,
            test_mean=test_mean,
            test_std=test_std,
            train_mean=train_mean,
            train_std=train_std,
            ref_test_mean=np.float64(ref_test),
            ref_train_mean=np.float64(ref_train),
            early_stopping_test_mean=np.float64(best_held_ref),
        )

    print(f"Wrote panel NPZs to: {out_dir}")
    print(f"Used alpha: {chosen_alpha} (requested {args.alpha})")
    print(f"Panels exported: {len(slugs)}")


if __name__ == "__main__":
    main()
