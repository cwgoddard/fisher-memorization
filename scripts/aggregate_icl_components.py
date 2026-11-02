#!/usr/bin/env python3
"""Average the component projection curves across seeds, one file per WSE alpha."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def natural_seed_key(path: Path) -> int:
    m = re.search(r"(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def alpha_suffix_to_float(s: str) -> float:
    return float(s.replace("p", "."))


def parse_group_meta(d: np.lib.npyio.NpzFile) -> tuple[list[str], list[str]]:
    if "_group_labels" not in d.files or "_group_slugs" not in d.files:
        raise ValueError("Missing _group_labels/_group_slugs in input npz.")
    labels = [str(x) for x in np.asarray(d["_group_labels"]).tolist()]
    slugs = [str(x) for x in np.asarray(d["_group_slugs"]).tolist()]
    if len(labels) != len(slugs):
        raise ValueError("Mismatched _group_labels and _group_slugs lengths.")
    return labels, slugs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", type=str, default="results/icl")
    parser.add_argument(
        "--input-prefix",
        type=str,
        default="fisher_component_projection_wse_a",
        help="Prefix before alpha suffix in each seed folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for per-alpha averaged outputs (default: <results-dir>/fisher_component_wse_alpha_means).",
    )
    parser.add_argument(
        "--alpha-suffixes",
        type=str,
        default="",
        help='Optional explicit suffix list, e.g. "0p0,0p1,0p2". Empty=auto-discover.',
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        print(f"Not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (results_dir / "fisher_component_wse_alpha_means").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(
        (p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")),
        key=natural_seed_key,
    )
    if not seed_dirs:
        print(f"No seed_* directories found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    if args.alpha_suffixes.strip():
        suffixes = [s.strip() for s in args.alpha_suffixes.split(",") if s.strip()]
    else:
        found: set[str] = set()
        pat = re.compile(rf"^{re.escape(args.input_prefix)}([0-9]+p[0-9]+|[0-9]+)\.npz$")
        for sd in seed_dirs:
            for fp in sd.glob(f"{args.input_prefix}*.npz"):
                m = pat.match(fp.name)
                if m:
                    found.add(m.group(1))
        suffixes = sorted(found, key=alpha_suffix_to_float)

    if not suffixes:
        print(
            f"No files matching {args.input_prefix}*.npz found under {results_dir}/seed_*/",
            file=sys.stderr,
        )
        sys.exit(1)

    for suffix in suffixes:
        fname = f"{args.input_prefix}{suffix}.npz"
        present = [sd for sd in seed_dirs if (sd / fname).is_file()]
        if not present:
            continue

        first = np.load(present[0] / fname, allow_pickle=True)
        labels, slugs = parse_group_meta(first)
        first.close()

        by_group = {slug: {"ks": None, "train": [], "test": []} for slug in slugs}
        refs_train = []
        refs_test = []
        used_seed_ids = []
        alpha_val = alpha_suffix_to_float(suffix)

        for sd in present:
            d = np.load(sd / fname, allow_pickle=True)
            cur_labels, cur_slugs = parse_group_meta(d)
            if cur_labels != labels or cur_slugs != slugs:
                raise ValueError(f"Group metadata mismatch in {sd / fname}")

            if "_ensemble_alpha" in d.files:
                cur_alpha = float(np.asarray(d["_ensemble_alpha"]).squeeze())
                if abs(cur_alpha - alpha_val) > 1e-9:
                    raise ValueError(f"Alpha mismatch in {sd / fname}: {cur_alpha} vs {alpha_val}")

            if "_ref_train" in d.files and "_ref_test" in d.files:
                refs_train.append(float(np.asarray(d["_ref_train"]).squeeze()))
                refs_test.append(float(np.asarray(d["_ref_test"]).squeeze()))

            for slug in slugs:
                ks = np.asarray(d[f"{slug}_ks"])
                tr = np.asarray(d[f"{slug}_train_losses"], dtype=np.float64)
                te = np.asarray(d[f"{slug}_test_losses"], dtype=np.float64)
                if by_group[slug]["ks"] is None:
                    by_group[slug]["ks"] = ks
                else:
                    ks_ref = by_group[slug]["ks"]
                    if ks_ref.shape != ks.shape or not np.allclose(ks_ref, ks):
                        raise ValueError(f"ks mismatch for {slug} in {sd / fname}")
                by_group[slug]["train"].append(tr)
                by_group[slug]["test"].append(te)
            used_seed_ids.append(natural_seed_key(sd))
            d.close()

        n_seeds = len(used_seed_ids)

        out: dict[str, np.ndarray] = {
            "_group_labels": np.array(labels, dtype=object),
            "_group_slugs": np.array(slugs, dtype=object),
            "_seed_ids": np.array(sorted(used_seed_ids), dtype=np.int32),
            "_n_seeds": np.array(n_seeds, dtype=np.int32),
            "_ensemble_alpha": np.array(alpha_val, dtype=np.float64),
        }

        if refs_train and refs_test:
            ref_train_mean = float(np.mean(refs_train))
            ref_test_mean = float(np.mean(refs_test))
            out["_ref_train_mean"] = np.array(ref_train_mean, dtype=np.float64)
            out["_ref_test_mean"] = np.array(ref_test_mean, dtype=np.float64)
        else:
            ref_train_mean = None
            ref_test_mean = None

        for i, (slug, label) in enumerate(zip(slugs, labels)):
            ks = by_group[slug]["ks"]
            train_stack = np.stack(by_group[slug]["train"], axis=0)
            test_stack = np.stack(by_group[slug]["test"], axis=0)

            train_mean = train_stack.mean(axis=0)
            train_std = (
                train_stack.std(axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(train_mean)
            )
            test_mean = test_stack.mean(axis=0)
            test_std = test_stack.std(axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(test_mean)

            out[f"{slug}_ks"] = ks
            out[f"{slug}_train_mean"] = train_mean
            out[f"{slug}_train_std"] = train_std
            out[f"{slug}_test_mean"] = test_mean
            out[f"{slug}_test_std"] = test_std

        out_npz = out_dir / f"{args.input_prefix}{suffix}_mean.npz"
        np.savez(out_npz, **out)
        print(f"Wrote {out_npz.name} " f"(alpha={alpha_val:.3f}, n_seeds={n_seeds})")

    print(f"All per-alpha means written under: {out_dir}")


if __name__ == "__main__":
    main()
