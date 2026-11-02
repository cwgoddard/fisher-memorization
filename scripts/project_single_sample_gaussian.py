"""Single-sample Fisher projections for the input-noise model (Fig. 5C, Tables 1-2)."""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
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
from src.gaussian_data import CIFAR10_TRANSFORM, TensorDatasetWithIntLabels
from src.gaussian_resnet import resnet20


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-sample Fisher projection for CIFAR-10 + Gaussian models"
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "experiments/resnet20_cifar10gaussian_g500_randlabels_run004"
        ),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=9,
        help="Number of top Fisher eigenvectors to project out; "
        "0 runs a matched BN-only control",
    )
    p.add_argument(
        "--num-cifar-fisher",
        type=int,
        default=1000,
        help="Number of CIFAR-10 train samples for Fisher computation",
    )
    p.add_argument(
        "--num-gaussian-fisher",
        type=int,
        default=1000,
        help="Max Gaussian samples for Fisher computation (capped by available)",
    )
    p.add_argument(
        "--num-eval-cifar-train",
        type=int,
        default=5000,
        help="CIFAR-10 train samples for bulk evaluation (disjoint from Fisher)",
    )
    p.add_argument(
        "--num-eval-class", type=int, default=1000, help="Per-class evaluation pool size"
    )
    p.add_argument(
        "--num-calib", type=int, default=4500, help="CIFAR-10 train samples for BN calibration"
    )
    p.add_argument(
        "--calib-batches",
        type=int,
        default=10,
        help="Max batches for BN recalibration (~2560 samples at bs=256)",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="CIFAR-10 data directory (default: read from model config)",
    )
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
        "'fullbn' writes gaussian_projection_results_fullbn_*",
    )
    p.add_argument(
        "--output-dir", type=str, default=None, help="Result directory (default: --model-path)"
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
    p.add_argument(
        "--save-eigenvectors",
        action="store_true",
        help="Save full eigenvectors (~10 MB per sample)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.top_k < 0:
        raise ValueError("--top-k must be nonnegative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file = os.path.join(args.model_path, "final_model.pt")
    config_file = os.path.join(args.model_path, "config.json")

    with open(config_file) as f:
        config = json.load(f)

    data_root = args.data_root or config.get("data_root", os.path.join(PROJECT_ROOT, "data"))

    print(f"Model:      {model_file}")
    print(f"Config:     {config.get('description', 'N/A')}")
    print(f"Data root:  {data_root}")
    print(f"Device:     {device}")

    model = resnet20(num_classes=10)
    load_model_weights(model_file, model, device=device)
    model.to(device).eval()

    all_layers = _get_targeted_layers(model)
    total_params = sum(e["n_params"] for e in all_layers)
    print(f"Conv/Linear layers: {len(all_layers)}, total weight params: {total_params}")

    cifar10_train = datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=CIFAR10_TRANSFORM
    )
    cifar10_test = datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=CIFAR10_TRANSFORM
    )

    gauss_meta = torch.load(
        os.path.join(args.model_path, "gaussian_metadata.pt"), weights_only=False
    )
    gauss_images = gauss_meta["images"]
    gauss_labels = gauss_meta["labels"]
    gauss_source_classes = gauss_meta["source_classes"]

    n_gaussian_total = len(gauss_images)
    n_gaussian_fisher = min(args.num_gaussian_fisher, n_gaussian_total)

    print(f"Gaussian set: {n_gaussian_total} total, " f"{n_gaussian_fisher} selected for Fisher")

    cifar_targets = torch.tensor(cifar10_train.targets)
    n_cifar = len(cifar10_train)

    all_cifar_perm = torch.randperm(n_cifar)
    ptr = 0

    cifar_fisher_indices = all_cifar_perm[ptr : ptr + args.num_cifar_fisher].numpy()
    ptr += args.num_cifar_fisher

    eval_cifar_indices = all_cifar_perm[ptr : ptr + args.num_eval_cifar_train].numpy()
    ptr += args.num_eval_cifar_train

    excluded_set = set(cifar_fisher_indices.tolist()) | set(eval_cifar_indices.tolist())

    calib_candidates = all_cifar_perm[ptr : ptr + args.num_calib].numpy()
    ptr += args.num_calib
    excluded_set |= set(calib_candidates.tolist())

    class_loaders = {}
    class_eval_indices = {}
    for c in range(10):
        class_indices = (cifar_targets == c).nonzero(as_tuple=True)[0].tolist()
        available = [idx for idx in class_indices if idx not in excluded_set]
        selected = available[: args.num_eval_class]
        if len(selected) < args.num_eval_class:
            print(f"  Warning: class {c} pool has {len(selected)}/{args.num_eval_class}")
        excluded_set |= set(selected)
        class_eval_indices[c] = np.asarray(selected, dtype=np.int64)
        class_loaders[c] = DataLoader(
            Subset(cifar10_train, selected),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

    gaussian_fisher_indices = torch.randperm(n_gaussian_total)[:n_gaussian_fisher].numpy()
    split_indices = {
        "fisher_cifar": np.asarray(cifar_fisher_indices, dtype=np.int64),
        "eval_cifar": np.asarray(eval_cifar_indices, dtype=np.int64),
        "calib_cifar": np.asarray(calib_candidates, dtype=np.int64),
        "fisher_gaussian": np.asarray(gaussian_fisher_indices, dtype=np.int64),
    }
    split_indices.update({f"eval_class_{c}": indices for c, indices in class_eval_indices.items()})

    print(
        f"CIFAR Fisher: {len(cifar_fisher_indices)}, "
        f"CIFAR eval: {len(eval_cifar_indices)}, "
        f"Calib: {len(calib_candidates)}"
    )
    print(f"Per-class pools: " f"{[len(class_loaders[c].dataset) for c in range(10)]}")

    eval_cifar_train_loader = DataLoader(
        Subset(cifar10_train, eval_cifar_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    cifar_test_loader = DataLoader(
        cifar10_test, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    gaussian_eval_loader = DataLoader(
        TensorDatasetWithIntLabels(gauss_images, gauss_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    calib_cifar_subset = Subset(cifar10_train, calib_candidates.tolist())
    calibration_dataset = ConcatDataset(
        [
            calib_cifar_subset,
            TensorDatasetWithIntLabels(gauss_images, gauss_labels),
        ]
    )
    calibration_loader = DataLoader(
        calibration_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    print(
        f"Calibration: {len(calib_candidates)} CIFAR + "
        f"{len(gauss_images)} Gaussian = {len(calibration_dataset)} samples, "
        f"{len(calibration_loader)} batches "
        f"(capped at {args.calib_batches})"
    )

    R = {
        "sample_type": [],  # 0=cifar, 1=gaussian
        "original_index": [],
        "assigned_label": [],
        "source_class": [],
        "eigenvalues": [],
        "self_prediction": [],
        "self_acc_assigned": [],
        "self_acc_groundtruth": [],
        "self_loss": [],
        "bulk_cifar_train_acc": [],
        "bulk_cifar_train_loss": [],
        "bulk_cifar_test_acc": [],
        "bulk_cifar_test_loss": [],
        "bulk_gaussian_acc": [],
        "bulk_gaussian_loss": [],
        "class_assigned_acc": [],
        "class_assigned_loss": [],
        "class_source_acc": [],
        "class_source_loss": [],
    }
    if args.save_eigenvectors:
        R["eigenvectors"] = []

    working_model = copy.deepcopy(model).to(device)
    original_w_flat = flatten_all_weights(model).to(device)
    working_layers = _get_targeted_layers(working_model)

    total_samples = len(cifar_fisher_indices) + n_gaussian_fisher
    save_dir = args.output_dir or args.model_path
    os.makedirs(save_dir, exist_ok=True)
    top_k = args.top_k
    tag_suffix = f"_{args.output_tag}" if args.output_tag else ""
    save_prefix = os.path.join(
        save_dir,
        f"gaussian_projection_results{tag_suffix}_numvectors_{top_k}",
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
        f"({len(cifar_fisher_indices)} CIFAR + {n_gaussian_fisher} Gaussian)"
    )
    print(f"Output prefix: {save_prefix}")
    print()

    t_total = time.time()

    for sample_num in range(total_samples):
        t_sample = time.time()

        if sample_num < len(cifar_fisher_indices):
            cidx = int(cifar_fisher_indices[sample_num])
            image, label = cifar10_train[cidx]
            source_class = label
            sample_type = 0
            original_index = cidx
        else:
            gi = sample_num - len(cifar_fisher_indices)
            g_idx = int(gaussian_fisher_indices[gi])
            image = gauss_images[g_idx]
            label = int(gauss_labels[g_idx].item())
            source_class = int(gauss_source_classes[g_idx].item())
            sample_type = 1
            original_index = g_idx

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

        ev_train = evaluate(working_model, eval_cifar_train_loader, device)
        ev_test = evaluate(working_model, cifar_test_loader, device)
        ev_gauss = evaluate(working_model, gaussian_eval_loader, device)

        working_model.eval()
        x = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
        with torch.no_grad():
            logits = working_model(x)
            pred = logits.argmax(dim=1).item()
            self_loss = F.cross_entropy(logits, torch.tensor([label], device=device)).item()

        ev_class_assigned = evaluate(working_model, class_loaders[label], device)
        if sample_type == 1 and source_class != label:
            ev_class_source = evaluate(working_model, class_loaders[source_class], device)
        else:
            ev_class_source = ev_class_assigned

        R["sample_type"].append(sample_type)
        R["original_index"].append(original_index)
        R["assigned_label"].append(label)
        R["source_class"].append(source_class)
        R["eigenvalues"].append(evals)
        R["self_prediction"].append(pred)
        R["self_acc_assigned"].append(float(pred == label))
        R["self_acc_groundtruth"].append(float(pred == source_class))
        R["self_loss"].append(self_loss)
        R["bulk_cifar_train_acc"].append(ev_train["accuracy"])
        R["bulk_cifar_train_loss"].append(ev_train["loss"])
        R["bulk_cifar_test_acc"].append(ev_test["accuracy"])
        R["bulk_cifar_test_loss"].append(ev_test["loss"])
        R["bulk_gaussian_acc"].append(ev_gauss["accuracy"])
        R["bulk_gaussian_loss"].append(ev_gauss["loss"])
        R["class_assigned_acc"].append(ev_class_assigned["accuracy"])
        R["class_assigned_loss"].append(ev_class_assigned["loss"])
        R["class_source_acc"].append(ev_class_source["accuracy"])
        R["class_source_loss"].append(ev_class_source["loss"])
        if args.save_eigenvectors:
            R["eigenvectors"].append(evecs)

        scatter_flat_weights(original_w_flat, working_layers)

        elapsed = time.time() - t_sample
        if sample_num < 5 or (sample_num + 1) % 10 == 0:
            tag = "CIFAR" if sample_type == 0 else "Gauss"
            print(
                f"  [{sample_num+1:>4d}/{total_samples}] {tag:5s} "
                f"idx={original_index:<5d} lbl={label} src={source_class} | "
                f"self={float(pred == label):.0f} "
                f"gt={float(pred == source_class):.0f} | "
                f"{elapsed:.1f}s"
            )

        if (sample_num + 1) % 100 == 0:
            _save_all(R, args, save_prefix, total_params, split_indices)

    elapsed_total = time.time() - t_total
    print(
        f"\nDone. {total_samples} samples in {elapsed_total:.0f}s "
        f"({elapsed_total / total_samples:.1f}s / sample)"
    )

    _save_all(R, args, save_prefix, total_params, split_indices)
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


def _save_all(R, args, prefix, total_params, split_indices):
    """Save results in .pt, .npz, and .json formats."""
    n = len(R["sample_type"])
    if n == 0:
        return

    pt_data = {
        "args": vars(args),
        "total_params": total_params,
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
    for key, values in split_indices.items():
        npz[f"split_{key}"] = np.asarray(values)
    np.savez(prefix + ".npz", **npz)

    st = np.array(R["sample_type"])
    cifar_mask = st == 0
    gauss_mask = st == 1

    def _agg(arr, mask):
        sub = np.array(arr)[mask]
        if len(sub) == 0:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(sub)), "std": float(np.std(sub)), "n": int(len(sub))}

    n_batches = getattr(args, "error_bar_batches", 20)
    summary = {
        "model_path": args.model_path,
        "total_samples": n,
        "n_cifar": int(cifar_mask.sum()),
        "n_gaussian": int(gauss_mask.sum()),
        "top_k": args.top_k,
        "total_params": total_params,
        "error_bar_batches": int(n_batches),
        "bn_averaging": args.bn_averaging,
        "num_calib_cifar": int(args.num_calib),
        "calibration_includes_all_gaussian": True,
        "batch_size": int(args.batch_size),
        "output_tag": args.output_tag,
        "cifar": {},
        "gaussian": {},
    }

    batched_metrics = {"self_acc_assigned", "self_acc_groundtruth"}
    for key in [
        "self_acc_assigned",
        "self_acc_groundtruth",
        "self_loss",
        "bulk_cifar_train_acc",
        "bulk_cifar_test_acc",
        "bulk_gaussian_acc",
        "class_assigned_acc",
        "class_source_acc",
    ]:
        agg_fn = (lambda a, m: _agg_batched(a, m, n_batches)) if key in batched_metrics else _agg
        summary["cifar"][key] = agg_fn(R[key], cifar_mask)
        summary["gaussian"][key] = agg_fn(R[key], gauss_mask)

    if R["eigenvalues"]:
        eig_arr = torch.stack(R["eigenvalues"]).numpy()
        if cifar_mask.any():
            summary["cifar"]["mean_eigenvalues"] = eig_arr[cifar_mask].mean(0).tolist()
        if gauss_mask.any():
            summary["gaussian"]["mean_eigenvalues"] = eig_arr[gauss_mask].mean(0).tolist()

    with open(prefix + ".json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
