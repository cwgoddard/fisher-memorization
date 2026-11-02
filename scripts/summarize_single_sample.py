"""Print Tables 1-2 and the Fig. 5C-D self-hit rates from the single-sample runs.

Reads the per-k JSON summaries written by project_single_sample_gaussian.py and
project_single_sample_label_noise.py, one file per value of k.
"""

import argparse
import json
from pathlib import Path

TABLE_K = (0, 1, 2, 4, 9)
SELF_HIT_K = tuple(range(10))
RESULT_ROOT = Path("sanity_checks/cumulative_bn_v1/single_sample")

GAUSSIAN_TABLE = [
    (
        "Fisher source CIFAR-10",
        "cifar",
        [
            ("CIFAR-10 train (5k samples)", "bulk_cifar_train_acc"),
            ("CIFAR-10 test", "bulk_cifar_test_acc"),
            ("Gaussian set", "bulk_gaussian_acc"),
            ("Same-class CIFAR (1k samples)", "class_assigned_acc"),
        ],
    ),
    (
        "Fisher source Gaussian",
        "gaussian",
        [
            ("CIFAR-10 train (5k samples)", "bulk_cifar_train_acc"),
            ("CIFAR-10 test", "bulk_cifar_test_acc"),
            ("Gaussian set", "bulk_gaussian_acc"),
            ("Assigned-label class (1k samples)", "class_assigned_acc"),
            ("Source-class CIFAR (1k samples)", "class_source_acc"),
        ],
    ),
]
NOISY_TABLE = [
    (
        "Fisher source uncorrupted",
        "uncorrupted",
        [
            ("CIFAR-10 train (5k samples)", "bulk_uncorrupted_acc"),
            ("CIFAR-10 test", "bulk_cifar_test_acc"),
            ("Corrupted CIFAR (assigned labels)", "bulk_corrupted_assigned_acc"),
            ("Corrupted CIFAR (true labels)", "bulk_corrupted_true_acc"),
            ("Same-class CIFAR (1k samples)", "class_assigned_acc"),
        ],
    ),
    (
        "Fisher source corrupted",
        "corrupted",
        [
            ("CIFAR-10 train (5k samples)", "bulk_uncorrupted_acc"),
            ("CIFAR-10 test", "bulk_cifar_test_acc"),
            ("Corrupted CIFAR (assigned labels)", "bulk_corrupted_assigned_acc"),
            ("Corrupted CIFAR (true labels)", "bulk_corrupted_true_acc"),
            ("Assigned-class CIFAR (1k samples)", "class_assigned_acc"),
            ("True-class CIFAR (1k samples)", "class_true_acc"),
        ],
    ),
]
GAUSSIAN_SELF_HIT = [
    ("CIFAR-10", "cifar", "self_acc_assigned"),
    ("Gaussian (assigned label)", "gaussian", "self_acc_assigned"),
    ("Gaussian (source label)", "gaussian", "self_acc_groundtruth"),
]
NOISY_SELF_HIT = [
    ("Uncorrupted", "uncorrupted", "self_acc_assigned"),
    ("Corrupted (assigned label)", "corrupted", "self_acc_assigned"),
    ("Corrupted (true label)", "corrupted", "self_acc_groundtruth"),
]


def load_summaries(result_dir, kind, tag, k_values):
    summaries = {}
    for k in k_values:
        path = Path(result_dir) / f"{kind}_projection_results_{tag}_numvectors_{k}.json"
        summaries[k] = json.loads(path.read_text())
        if summaries[k]["top_k"] != k:
            raise ValueError(f"{path} records top_k={summaries[k]['top_k']}, expected {k}")
    return summaries


def print_bulk_table(title, summaries, table):
    print(f"\n{title}: bulk accuracy (%) after sample projection")
    print(f"{'':42s}" + "".join(f"{f'k={k}':>8s}" for k in TABLE_K))
    for source, group, rows in table:
        print(source)
        for label, metric in rows:
            values = [100 * summaries[k][group][metric]["mean"] for k in TABLE_K]
            print(f"  {label:40s}" + "".join(f"{value:8.1f}" for value in values))


def print_self_hit(title, summaries, series):
    print(f"\n{title}: self-hit rate, mean +/- SD over {summaries[0]['error_bar_batches']} batches")
    print(f"{'':30s}" + "".join(f"{f'k={k}':>14s}" for k in SELF_HIT_K))
    for label, group, metric in series:
        cells = [summaries[k][group][metric] for k in SELF_HIT_K]
        print(f"{label:30s}" + "".join(f"{c['mean']:.3f}+-{c['std']:.3f}".rjust(14) for c in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gaussian-dir", type=str, default=str(RESULT_ROOT / "gaussian"))
    parser.add_argument("--noisy-dir", type=str, default=str(RESULT_ROOT / "noisy"))
    parser.add_argument("--tag", type=str, default="cumulative_bn_v1")
    args = parser.parse_args()

    gaussian = load_summaries(args.gaussian_dir, "gaussian", args.tag, SELF_HIT_K)
    noisy = load_summaries(args.noisy_dir, "noisy", args.tag, SELF_HIT_K)

    print_bulk_table("Gaussian model", gaussian, GAUSSIAN_TABLE)
    print_bulk_table("Noisy-label model", noisy, NOISY_TABLE)
    print_self_hit("Gaussian model", gaussian, GAUSSIAN_SELF_HIT)
    print_self_hit("Noisy-label model", noisy, NOISY_SELF_HIT)


if __name__ == "__main__":
    main()
