"""Single-sample Fisher projections for the label-noise model (Fig. 5D, Tables 1-2)."""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from _bootstrap import PROJECT_ROOT
from src.experiment_utils import evaluate, load_model_weights, save_experiment_data
from src.fisher_projection import (
    _get_targeted_layers,
    _recalibrate_batchnorm,
    flatten_all_weights,
    full_model_single_sample_fisher_top_k,
    scatter_flat_weights,
)
from src.gaussian_data import CIFAR10_TRANSFORM
from src.gaussian_resnet import resnet20


class RelabeledSubset(Dataset):
    """Wraps a base dataset + index list, overriding labels from an array."""

    def __init__(self, base_dataset, indices, label_array):
        self.base = base_dataset
        self.indices = list(indices)
        self.label_array = label_array

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, _ = self.base[real_idx]
        return image, int(self.label_array[real_idx])


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-sample Fisher projection for label-noise models"
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "output/noisy10pct"),
        help="Path to noisy-label model directory",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=9,
        help="Number of top Fisher eigenvectors to project out; "
        "0 runs a matched BN-only control",
    )
    p.add_argument(
        "--num-fisher",
        type=int,
        default=1000,
        help="Number of Fisher samples per category " "(corrupted / uncorrupted)",
    )
    p.add_argument(
        "--num-eval-uncorrupted",
        type=int,
        default=5000,
        help="Uncorrupted train samples for bulk evaluation",
    )
    p.add_argument(
        "--num-eval-corrupted",
        type=int,
        default=5000,
        help="Max corrupted train samples for bulk evaluation",
    )
    p.add_argument(
        "--num-eval-class",
        type=int,
        default=1000,
        help="Per-class evaluation pool size (from uncorrupted train)",
    )
    p.add_argument(
        "--num-calib", type=int, default=4500, help="CIFAR-10 train samples for BN calibration"
    )
    p.add_argument("--calib-batches", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-root", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--error-bar-batches",
        type=int,
        default=20,
        help="Number of equal-sized batches used to compute the "
        "self-hit-rate error bar (mean/STD across batch means)",
    )
    p.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional tag inserted into result filenames, e.g. "
        "'fullbn' writes noisy_projection_results_fullbn_*",
    )
    p.add_argument(
        "--output-dir", type=str, default=None, help="Result directory (default: --model-dir)"
    )
    p.add_argument(
        "--calib-source",
        choices=("clean", "all"),
        default="clean",
        help="Select BN calibration images from clean examples only "
        "or from all remaining training examples",
    )
    p.add_argument(
        "--bn-averaging",
        choices=("exponential", "cumulative"),
        default="cumulative",
        help="BatchNorm running-stat averaging during recalibration",
    )
    p.add_argument(
        "--fail-if-output-exists",
        action="store_true",
        help="Refuse to overwrite an existing result triplet",
    )
    p.add_argument("--save-eigenvectors", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.top_k < 0:
        raise ValueError("--top-k must be nonnegative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file = os.path.join(args.model_dir, "model.pt")
    print(f"Model:      {model_file}")
    print(f"Device:     {device}")

    model = resnet20(num_classes=10)
    load_model_weights(model_file, model, device=device)
    model.to(device).eval()

    all_layers = _get_targeted_layers(model)
    total_params = sum(e["n_params"] for e in all_layers)
    print(f"Conv/Linear layers: {len(all_layers)}, " f"total weight params: {total_params}")

    meta_path = os.path.join(args.model_dir, "noise_metadata.npz")
    noise_meta = np.load(meta_path, allow_pickle=True)

    true_labels = noise_meta["true_labels"]  # (50000,) correct labels
    noised_labels = noise_meta["noised_labels"]  # (50000,) training labels
    is_noised = noise_meta["is_noised"]  # (50000,) bool
    noise_fraction = float(noise_meta["noise_fraction"])

    corrupted_all = np.where(is_noised)[0]
    clean_all = np.where(~is_noised)[0]
    print(f"Noise fraction: {noise_fraction:.0%}")
    print(f"Corrupted: {len(corrupted_all)}, Clean: {len(clean_all)}")

    cifar10_train = datasets.CIFAR10(
        root=args.data_root, train=True, download=False, transform=CIFAR10_TRANSFORM
    )
    cifar10_test = datasets.CIFAR10(
        root=args.data_root, train=False, download=False, transform=CIFAR10_TRANSFORM
    )

    rng = np.random.RandomState(args.seed)

    if args.calib_source == "all":
        n_calib_corrupted = int(round(args.num_calib * noise_fraction))
        n_calib_clean = args.num_calib - n_calib_corrupted
        if n_calib_corrupted > len(corrupted_all) or n_calib_clean > len(clean_all):
            raise ValueError("Requested stratified calibration set is too large")
        calib_corrupted = rng.choice(corrupted_all, size=n_calib_corrupted, replace=False)
        calib_clean = rng.choice(clean_all, size=n_calib_clean, replace=False)
        calib_indices = np.concatenate((calib_clean, calib_corrupted))
        rng.shuffle(calib_indices)
        reserved_calib_set = set(calib_indices.tolist())
    else:
        calib_clean = np.empty(0, dtype=np.int64)
        calib_corrupted = np.empty(0, dtype=np.int64)
        calib_indices = np.empty(0, dtype=np.int64)
        reserved_calib_set = set()

    fisher_corrupted_pool = np.array([i for i in corrupted_all if i not in reserved_calib_set])
    fisher_clean_pool = np.array([i for i in clean_all if i not in reserved_calib_set])
    n_fisher_corrupted = min(args.num_fisher, len(fisher_corrupted_pool))
    n_fisher_clean = min(args.num_fisher, len(fisher_clean_pool))

    fisher_corrupted = rng.choice(fisher_corrupted_pool, size=n_fisher_corrupted, replace=False)
    fisher_clean = rng.choice(fisher_clean_pool, size=n_fisher_clean, replace=False)

    fisher_set = set(fisher_corrupted.tolist()) | set(fisher_clean.tolist())

    remaining_clean = np.array(
        [i for i in clean_all if i not in fisher_set and i not in reserved_calib_set]
    )
    rng.shuffle(remaining_clean)

    eval_uncorrupted_indices = remaining_clean[: args.num_eval_uncorrupted]

    remaining_corrupted = np.array(
        [i for i in corrupted_all if i not in fisher_set and i not in reserved_calib_set]
    )
    n_eval_corrupted = min(args.num_eval_corrupted, len(remaining_corrupted))
    eval_corrupted_indices = rng.choice(remaining_corrupted, size=n_eval_corrupted, replace=False)

    excluded_set = (
        fisher_set
        | set(eval_uncorrupted_indices.tolist())
        | set(eval_corrupted_indices.tolist())
        | reserved_calib_set
    )

    if args.calib_source == "clean":
        calib_pool = np.array([i for i in clean_all if i not in excluded_set])
        rng.shuffle(calib_pool)
        calib_indices = calib_pool[: args.num_calib]
        calib_clean = calib_indices.copy()
    excluded_set |= set(calib_indices.tolist())

    class_loaders = {}
    class_eval_indices = {}
    for c in range(10):
        class_pool = np.array(
            [i for i in clean_all if true_labels[i] == c and i not in excluded_set]
        )
        selected = class_pool[: args.num_eval_class]
        if len(selected) < args.num_eval_class:
            print(f"  Warning: class {c} pool has " f"{len(selected)}/{args.num_eval_class}")
        excluded_set |= set(selected.tolist())
        class_eval_indices[c] = np.asarray(selected, dtype=np.int64)
        class_loaders[c] = DataLoader(
            Subset(cifar10_train, selected.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
    split_indices = {
        "fisher_clean": np.asarray(fisher_clean, dtype=np.int64),
        "fisher_corrupted": np.asarray(fisher_corrupted, dtype=np.int64),
        "eval_clean": np.asarray(eval_uncorrupted_indices, dtype=np.int64),
        "eval_corrupted": np.asarray(eval_corrupted_indices, dtype=np.int64),
        "calib_cifar": np.asarray(calib_indices, dtype=np.int64),
        "calib_clean": np.asarray(calib_clean, dtype=np.int64),
        "calib_corrupted": np.asarray(calib_corrupted, dtype=np.int64),
    }
    split_indices.update({f"eval_class_{c}": indices for c, indices in class_eval_indices.items()})

    print(f"Fisher: {n_fisher_corrupted} corrupted + " f"{n_fisher_clean} uncorrupted")
    print(f"Eval: {len(eval_uncorrupted_indices)} uncorrupted, " f"{n_eval_corrupted} corrupted")
    print(
        f"Calib: {len(calib_indices)} samples "
        f"({len(calib_clean)} clean + {len(calib_corrupted)} corrupted; "
        f"source={args.calib_source})"
    )
    print(f"Per-class pools: " f"{[len(class_loaders[c].dataset) for c in range(10)]}")

    cifar_test_loader = DataLoader(
        cifar10_test, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    eval_uncorrupted_loader = DataLoader(
        Subset(cifar10_train, eval_uncorrupted_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    corrupted_assigned_loader = DataLoader(
        RelabeledSubset(cifar10_train, eval_corrupted_indices, noised_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    corrupted_true_loader = DataLoader(
        Subset(cifar10_train, eval_corrupted_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    calibration_loader = DataLoader(
        Subset(cifar10_train, calib_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    print(f"Calibration: {len(calib_indices)} samples, " f"capped at {args.calib_batches} batches")

    R = {
        "sample_type": [],  # 0=uncorrupted, 1=corrupted
        "cifar_index": [],
        "assigned_label": [],
        "true_label": [],
        "eigenvalues": [],
        "self_prediction": [],
        "self_acc_assigned": [],
        "self_acc_groundtruth": [],
        "self_loss": [],
        "bulk_cifar_test_acc": [],
        "bulk_cifar_test_loss": [],
        "bulk_uncorrupted_acc": [],
        "bulk_uncorrupted_loss": [],
        "bulk_corrupted_assigned_acc": [],
        "bulk_corrupted_assigned_loss": [],
        "bulk_corrupted_true_acc": [],
        "bulk_corrupted_true_loss": [],
        "class_assigned_acc": [],
        "class_assigned_loss": [],
        "class_true_acc": [],
        "class_true_loss": [],
    }
    if args.save_eigenvectors:
        R["eigenvectors"] = []

    working_model = copy.deepcopy(model).to(device)
    original_w_flat = flatten_all_weights(model).to(device)
    working_layers = _get_targeted_layers(working_model)

    total_samples = n_fisher_clean + n_fisher_corrupted
    save_dir = args.output_dir or args.model_dir
    os.makedirs(save_dir, exist_ok=True)
    top_k = args.top_k
    tag_suffix = f"_{args.output_tag}" if args.output_tag else ""
    save_prefix = os.path.join(
        save_dir,
        f"noisy_projection_results{tag_suffix}_numvectors_{top_k}",
    )
    if args.fail_if_output_exists:
        existing = [
            save_prefix + extension
            for extension in (".pt", ".npz", ".json")
            if os.path.exists(save_prefix + extension)
        ]
        if existing:
            raise FileExistsError("Refusing to overwrite existing results: " + ", ".join(existing))

    print(
        f"\nTotal Fisher samples: {total_samples} "
        f"({n_fisher_clean} uncorrupted + {n_fisher_corrupted} corrupted)"
    )
    print(f"Output prefix: {save_prefix}")
    print()

    t_total = time.time()

    # Uncorrupted first, then corrupted
    all_fisher_items = []
    for ci in fisher_clean:
        all_fisher_items.append((0, int(ci)))  # (sample_type, cifar_idx)
    for ci in fisher_corrupted:
        all_fisher_items.append((1, int(ci)))

    for sample_num, (sample_type, cifar_idx) in enumerate(all_fisher_items):
        t_sample = time.time()

        image, _ = cifar10_train[cifar_idx]
        true_lab = int(true_labels[cifar_idx])
        assigned_lab = int(noised_labels[cifar_idx])

        if sample_type == 0:
            label_for_self = true_lab
        else:
            label_for_self = assigned_lab

        if args.top_k == 0:
            evals = torch.empty(0)
            evecs = torch.empty((original_w_flat.numel(), 0))
        else:
            evals, evecs = full_model_single_sample_fisher_top_k(model, image, device, k=args.top_k)

        evecs_gpu = evecs.to(device=device, dtype=original_w_flat.dtype)
        if args.top_k == 0:
            w_projected = original_w_flat
        else:
            coeffs = evecs_gpu.T @ original_w_flat
            w_projected = original_w_flat - evecs_gpu @ coeffs
        scatter_flat_weights(w_projected, working_layers)

        _recalibrate_batchnorm(
            working_model,
            calibration_loader,
            device,
            max_batches=args.calib_batches,
            cumulative=args.bn_averaging == "cumulative",
        )

        ev_test = evaluate(working_model, cifar_test_loader, device)
        ev_uncorr = evaluate(working_model, eval_uncorrupted_loader, device)
        ev_corr_assigned = evaluate(working_model, corrupted_assigned_loader, device)
        ev_corr_true = evaluate(working_model, corrupted_true_loader, device)

        working_model.eval()
        x = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
        with torch.no_grad():
            logits = working_model(x)
            pred = logits.argmax(dim=1).item()
            self_loss = F.cross_entropy(
                logits, torch.tensor([label_for_self], device=device)
            ).item()

        ev_class_assigned = evaluate(working_model, class_loaders[assigned_lab], device)
        if assigned_lab != true_lab:
            ev_class_true = evaluate(working_model, class_loaders[true_lab], device)
        else:
            ev_class_true = ev_class_assigned

        R["sample_type"].append(sample_type)
        R["cifar_index"].append(cifar_idx)
        R["assigned_label"].append(assigned_lab)
        R["true_label"].append(true_lab)
        R["eigenvalues"].append(evals)
        R["self_prediction"].append(pred)
        R["self_acc_assigned"].append(float(pred == assigned_lab))
        R["self_acc_groundtruth"].append(float(pred == true_lab))
        R["self_loss"].append(self_loss)
        R["bulk_cifar_test_acc"].append(ev_test["accuracy"])
        R["bulk_cifar_test_loss"].append(ev_test["loss"])
        R["bulk_uncorrupted_acc"].append(ev_uncorr["accuracy"])
        R["bulk_uncorrupted_loss"].append(ev_uncorr["loss"])
        R["bulk_corrupted_assigned_acc"].append(ev_corr_assigned["accuracy"])
        R["bulk_corrupted_assigned_loss"].append(ev_corr_assigned["loss"])
        R["bulk_corrupted_true_acc"].append(ev_corr_true["accuracy"])
        R["bulk_corrupted_true_loss"].append(ev_corr_true["loss"])
        R["class_assigned_acc"].append(ev_class_assigned["accuracy"])
        R["class_assigned_loss"].append(ev_class_assigned["loss"])
        R["class_true_acc"].append(ev_class_true["accuracy"])
        R["class_true_loss"].append(ev_class_true["loss"])
        if args.save_eigenvectors:
            R["eigenvectors"].append(evecs)

        scatter_flat_weights(original_w_flat, working_layers)

        elapsed = time.time() - t_sample
        if sample_num < 5 or (sample_num + 1) % 10 == 0:
            tag = "clean" if sample_type == 0 else "noisy"
            print(
                f"  [{sample_num+1:>4d}/{total_samples}] {tag:5s} "
                f"idx={cifar_idx:<5d} assigned={assigned_lab} "
                f"true={true_lab} | "
                f"self_a={float(pred == assigned_lab):.0f} "
                f"self_t={float(pred == true_lab):.0f} | "
                f"{elapsed:.1f}s"
            )

        if (sample_num + 1) % 100 == 0:
            _save_all(R, args, save_prefix, total_params, noise_fraction, split_indices)

    elapsed_total = time.time() - t_total
    print(
        f"\nDone. {total_samples} samples in {elapsed_total:.0f}s "
        f"({elapsed_total / total_samples:.1f}s / sample)"
    )

    _save_all(R, args, save_prefix, total_params, noise_fraction, split_indices)
    print(f"Results saved to {save_prefix}.{{pt,npz,json}}")


def _agg_batched(arr, mask, n_batches, seed=0):
    sub = np.array(arr)[mask]
    n = len(sub)
    if n == 0:
        return {"mean": None, "std": None, "n": 0, "n_batches": 0}
    nb = int(min(n_batches, n))
    rng = np.random.RandomState(seed)
    sub = sub[rng.permutation(n)]
    batch_means = np.array([b.mean() for b in np.array_split(sub, nb)])
    return {
        "mean": float(batch_means.mean()),
        "std": float(batch_means.std()),
        "n": int(n),
        "n_batches": int(nb),
    }


def _save_all(R, args, prefix, total_params, noise_fraction, split_indices):
    n = len(R["sample_type"])
    if n == 0:
        return

    pt_data = {
        "args": vars(args),
        "total_params": total_params,
        "noise_fraction": noise_fraction,
        "n_samples": n,
        "split_indices": split_indices,
    }
    for key in R:
        if key == "eigenvalues":
            pt_data[key] = torch.stack(R[key]) if R[key] else torch.empty(0)
        elif key == "eigenvectors":
            pt_data[key] = R[key]
        else:
            pt_data[key] = R[key]
    save_experiment_data(pt_data, prefix + ".pt")

    npz = {}
    for key in R:
        if key == "eigenvectors":
            continue
        vals = R[key]
        if key == "eigenvalues":
            npz[key] = torch.stack(vals).numpy() if vals else np.empty((0, args.top_k))
        else:
            npz[key] = np.array(vals)
    npz["fisher_clean_indices"] = split_indices["fisher_clean"]
    npz["fisher_corrupted_indices"] = split_indices["fisher_corrupted"]
    for key, values in split_indices.items():
        npz[f"split_{key}"] = np.asarray(values)
    np.savez(prefix + ".npz", **npz)

    st = np.array(R["sample_type"])
    clean_mask = st == 0
    corr_mask = st == 1

    def _agg(arr, mask):
        sub = np.array(arr)[mask]
        if len(sub) == 0:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(sub)), "std": float(np.std(sub)), "n": int(len(sub))}

    n_batches = getattr(args, "error_bar_batches", 20)
    summary = {
        "model_dir": args.model_dir,
        "noise_fraction": noise_fraction,
        "total_samples": n,
        "n_uncorrupted": int(clean_mask.sum()),
        "n_corrupted": int(corr_mask.sum()),
        "top_k": args.top_k,
        "total_params": total_params,
        "error_bar_batches": int(n_batches),
        "bn_averaging": args.bn_averaging,
        "num_calib_cifar": int(args.num_calib),
        "num_calib_clean": int(len(split_indices["calib_clean"])),
        "num_calib_corrupted": int(len(split_indices["calib_corrupted"])),
        "calib_source": args.calib_source,
        "batch_size": int(args.batch_size),
        "output_tag": args.output_tag,
        "uncorrupted": {},
        "corrupted": {},
    }
    # Metrics whose error bar is computed by batching (per-sample 0/1 self-hits):
    # partition the population into `n_batches` equal batches, take the per-batch
    # self-hit rate, and report mean/STD across batch means.
    batched_metrics = {"self_acc_assigned", "self_acc_groundtruth"}
    for key in [
        "self_acc_assigned",
        "self_acc_groundtruth",
        "self_loss",
        "bulk_cifar_test_acc",
        "bulk_uncorrupted_acc",
        "bulk_corrupted_assigned_acc",
        "bulk_corrupted_true_acc",
        "class_assigned_acc",
        "class_true_acc",
    ]:
        agg_fn = (lambda a, m: _agg_batched(a, m, n_batches)) if key in batched_metrics else _agg
        summary["uncorrupted"][key] = agg_fn(R[key], clean_mask)
        summary["corrupted"][key] = agg_fn(R[key], corr_mask)

    if R["eigenvalues"]:
        eig_arr = torch.stack(R["eigenvalues"]).numpy()
        if clean_mask.any():
            summary["uncorrupted"]["mean_eigenvalues"] = eig_arr[clean_mask].mean(0).tolist()
        if corr_mask.any():
            summary["corrupted"]["mean_eigenvalues"] = eig_arr[corr_mask].mean(0).tolist()

    with open(prefix + ".json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
