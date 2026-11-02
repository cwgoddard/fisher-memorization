"""Matched unchanged-weight controls for cumulative BatchNorm calibration.

This script is intentionally lightweight relative to the projection sweeps. It
checks whether population metrics and BN running statistics have stabilized
after 1, 2, 5, 10, and all calibration batches. No Fisher matrix is computed.
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets

from _bootstrap import PROJECT_ROOT
from src.experiment_utils import evaluate, load_model_weights
from src.fisher_projection import _recalibrate_batchnorm
from src.gaussian_data import CIFAR10_TRANSFORM, TensorDatasetWithIntLabels
from src.gaussian_resnet import resnet20


class RelabeledSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, labels):
        self.base = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_index = int(self.indices[index])
        image, _ = self.base[real_index]
        return image, int(self.labels[real_index])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("gaussian", "noisy"), required=True)
    parser.add_argument(
        "--gaussian-model-path",
        type=str,
        default=os.path.join(
            PROJECT_ROOT,
            "experiments/resnet20_cifar10gaussian_g500_randlabels_run004",
        ),
    )
    parser.add_argument(
        "--noisy-model-dir", type=str, default=os.path.join(PROJECT_ROOT, "output/noisy10pct")
    )
    parser.add_argument("--data-root", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--output-tag", type=str, default="cumulative_bn_v1")
    parser.add_argument("--num-calib", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--batch-counts", type=str, default="1,2,5,10,all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-order-seed", type=int, default=1042)
    parser.add_argument("--fail-if-output-exists", action="store_true")
    return parser.parse_args()


def parse_batch_counts(spec, total_batches):
    counts = []
    for token in spec.split(","):
        token = token.strip().lower()
        count = total_batches if token == "all" else int(token)
        if count <= 0:
            raise ValueError("Calibration batch counts must be positive")
        count = min(count, total_batches)
        if count not in counts:
            counts.append(count)
    if total_batches not in counts:
        counts.append(total_batches)
    return counts


def batchnorm_state(model):
    means = []
    variances = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            means.append(module.running_mean.detach().cpu().flatten())
            variances.append(module.running_var.detach().cpu().flatten())
    return {
        "running_mean": torch.cat(means),
        "running_var": torch.cat(variances),
    }


def make_gaussian_problem(args):
    model_path = args.gaussian_model_path
    train = datasets.CIFAR10(
        root=args.data_root, train=True, download=False, transform=CIFAR10_TRANSFORM
    )
    test = datasets.CIFAR10(
        root=args.data_root, train=False, download=False, transform=CIFAR10_TRANSFORM
    )
    gaussian_metadata = torch.load(os.path.join(model_path, "gaussian_metadata.pt"))
    gauss_images = gaussian_metadata["images"]
    gauss_labels = gaussian_metadata["labels"]

    permutation = torch.randperm(len(train))
    fisher_indices = permutation[:1000].numpy()
    eval_indices = permutation[1000:6000].numpy()
    calib_indices = permutation[6000 : 6000 + args.num_calib].numpy()

    calibration_dataset = ConcatDataset(
        [
            Subset(train, calib_indices.tolist()),
            TensorDatasetWithIntLabels(gauss_images, gauss_labels),
        ]
    )
    eval_loaders = {
        "cifar_train": DataLoader(
            Subset(train, eval_indices.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "cifar_test": DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0),
        "gaussian": DataLoader(
            TensorDatasetWithIntLabels(gauss_images, gauss_labels),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }
    split_indices = {
        "fisher_cifar": fisher_indices,
        "eval_cifar": eval_indices,
        "calib_cifar": calib_indices,
    }
    return (
        os.path.join(model_path, "final_model.pt"),
        calibration_dataset,
        eval_loaders,
        split_indices,
        {"num_calib_gaussian": len(gauss_images)},
    )


def make_noisy_problem(args):
    model_dir = args.noisy_model_dir
    train = datasets.CIFAR10(
        root=args.data_root, train=True, download=False, transform=CIFAR10_TRANSFORM
    )
    test = datasets.CIFAR10(
        root=args.data_root, train=False, download=False, transform=CIFAR10_TRANSFORM
    )
    noise_metadata = np.load(os.path.join(model_dir, "noise_metadata.npz"), allow_pickle=True)
    true_labels = noise_metadata["true_labels"]
    noised_labels = noise_metadata["noised_labels"]
    is_noised = noise_metadata["is_noised"]
    corrupted_all = np.where(is_noised)[0]
    clean_all = np.where(~is_noised)[0]

    rng = np.random.RandomState(args.seed)
    noise_fraction = float(np.mean(is_noised))
    n_calib_corrupted = int(round(args.num_calib * noise_fraction))
    n_calib_clean = args.num_calib - n_calib_corrupted
    calib_corrupted = rng.choice(corrupted_all, size=n_calib_corrupted, replace=False)
    calib_clean = rng.choice(clean_all, size=n_calib_clean, replace=False)
    calib_indices = np.concatenate((calib_clean, calib_corrupted))
    rng.shuffle(calib_indices)
    calib_set = set(calib_indices.tolist())

    fisher_corrupted_pool = np.array([i for i in corrupted_all if i not in calib_set])
    fisher_clean_pool = np.array([i for i in clean_all if i not in calib_set])
    fisher_corrupted = rng.choice(
        fisher_corrupted_pool,
        size=min(1000, len(fisher_corrupted_pool)),
        replace=False,
    )
    fisher_clean = rng.choice(
        fisher_clean_pool,
        size=min(1000, len(fisher_clean_pool)),
        replace=False,
    )
    fisher_set = set(fisher_corrupted.tolist()) | set(fisher_clean.tolist())

    remaining_clean = np.array([i for i in clean_all if i not in fisher_set and i not in calib_set])
    rng.shuffle(remaining_clean)
    eval_clean = remaining_clean[:5000]

    remaining_corrupted = np.array(
        [i for i in corrupted_all if i not in fisher_set and i not in calib_set]
    )
    eval_corrupted = rng.choice(
        remaining_corrupted,
        size=min(5000, len(remaining_corrupted)),
        replace=False,
    )
    calibration_dataset = Subset(train, calib_indices.tolist())
    eval_loaders = {
        "cifar_test": DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0),
        "uncorrupted_train": DataLoader(
            Subset(train, eval_clean.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "corrupted_assigned": DataLoader(
            RelabeledSubset(train, eval_corrupted, noised_labels),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "corrupted_true": DataLoader(
            RelabeledSubset(train, eval_corrupted, true_labels),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }
    split_indices = {
        "fisher_clean": fisher_clean,
        "fisher_corrupted": fisher_corrupted,
        "eval_clean": eval_clean,
        "eval_corrupted": eval_corrupted,
        "calib_cifar": calib_indices,
        "calib_clean": calib_clean,
        "calib_corrupted": calib_corrupted,
    }
    return (
        os.path.join(model_dir, "model.pt"),
        calibration_dataset,
        eval_loaders,
        split_indices,
        {
            "calib_source": "all",
            "num_calib_clean": len(calib_clean),
            "num_calib_corrupted": len(calib_corrupted),
        },
    )


def main():
    args = parse_args()
    if args.num_calib <= 0 or args.batch_size <= 0:
        raise ValueError("--num-calib and --batch-size must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = resnet20(num_classes=10)
    problem = (
        make_gaussian_problem(args) if args.experiment == "gaussian" else make_noisy_problem(args)
    )
    model_file, calibration_dataset, eval_loaders, split_indices, extra = problem
    load_model_weights(model_file, base_model, device=device)
    base_model.to(device).eval()

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    total_batches = len(calibration_loader)
    batch_counts = parse_batch_counts(args.batch_counts, total_batches)

    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"bn_convergence_{args.experiment}_{args.output_tag}"
    prefix = os.path.join(args.output_dir, stem)
    output_paths = [prefix + ".pt", prefix + ".json"]
    if args.fail_if_output_exists:
        existing = [path for path in output_paths if os.path.exists(path)]
        if existing:
            raise FileExistsError("Refusing to overwrite existing controls: " + ", ".join(existing))

    records = {}
    states = {}
    for count in batch_counts:
        model = copy.deepcopy(base_model).to(device)
        torch.manual_seed(args.calibration_order_seed)
        _recalibrate_batchnorm(
            model,
            calibration_loader,
            device,
            max_batches=count,
            cumulative=True,
        )
        states[count] = batchnorm_state(model)
        records[count] = {
            name: evaluate(model, loader, device) for name, loader in eval_loaders.items()
        }
        print(f"Completed {count}/{total_batches} calibration batches")

    full_state = states[total_batches]
    state_distances = {}
    for count, state in states.items():
        state_distances[count] = {
            "running_mean_l2_to_all": float(
                torch.linalg.vector_norm(state["running_mean"] - full_state["running_mean"])
            ),
            "running_var_l2_to_all": float(
                torch.linalg.vector_norm(state["running_var"] - full_state["running_var"])
            ),
        }

    metadata = {
        "experiment": args.experiment,
        "model_file": model_file,
        "num_calib_cifar": args.num_calib,
        "num_calib_gaussian": extra.get("num_calib_gaussian", 0),
        "num_calib_clean": extra.get("num_calib_clean"),
        "num_calib_corrupted": extra.get("num_calib_corrupted"),
        "calib_source": extra.get("calib_source", "held-out CIFAR + all Gaussian"),
        "batch_size": args.batch_size,
        "total_calibration_batches": total_batches,
        "batch_counts": batch_counts,
        "bn_averaging": "cumulative",
        "seed": args.seed,
        "calibration_order_seed": args.calibration_order_seed,
        "output_tag": args.output_tag,
    }
    torch.save(
        {
            "metadata": metadata,
            "records": records,
            "batchnorm_states": states,
            "state_distances": state_distances,
            "split_indices": split_indices,
        },
        output_paths[0],
    )

    json_records = {}
    for count, evaluations in records.items():
        json_records[str(count)] = {
            "evaluations": evaluations,
            **state_distances[count],
        }
    with open(output_paths[1], "w") as handle:
        json.dump(
            {
                "metadata": metadata,
                "records": json_records,
            },
            handle,
            indent=2,
        )
    print(f"Saved convergence controls to {prefix}.{{pt,json}}")


if __name__ == "__main__":
    main()
