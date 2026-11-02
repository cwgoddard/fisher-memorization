"""Random-orthonormal control for the bulk layerwise Fisher experiment.

This script mirrors ``project_gaussian_layerwise.py`` but never computes or loads
Fisher information.  It draws one Haar-random orthonormal basis for the
selected layer, randomly orders that basis, and uses the first k directions as
the nested projection subspace at each evaluated value of k.
"""

import argparse
import copy
import os
import re
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision import datasets

from _bootstrap import PROJECT_ROOT
from src.experiment_utils import evaluate, save_experiment_data
from src.fisher_projection import _get_targeted_layers, _recalibrate_batchnorm
from src.gaussian_data import CIFAR10_TRANSFORM, TensorDatasetWithIntLabels
from src.gaussian_resnet import resnet20

DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "experiments/resnet20_cifar10gaussian_g500_randlabels_run004",
)
NUM_WORKERS = 0
BATCH_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bulk projection experiment using a random orthonormal basis."
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--scope",
        type=str,
        default="first",
        help=(
            "Layer to target: 'first', 'last', an integer depth index, or a "
            "layer name (use --list-layers to see options)"
        ),
    )
    parser.add_argument(
        "--num-k-points",
        type=int,
        default=None,
        help="Cap on the number of k-values to evaluate (uniformly spaced). Default: all.",
    )
    parser.add_argument(
        "--num-calib-cifar",
        type=int,
        default=4500,
        help=(
            "Number of CIFAR train samples in the BN calibration set "
            "(all Gaussian images are always included)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for the basis and data sampling. By default, use the "
            "auto-incremented experiment run number."
        ),
    )
    parser.add_argument(
        "--no-save-basis",
        action="store_true",
        help=(
            "Do not save the dense random basis. The seed and column "
            "permutation are still saved with the metrics."
        ),
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print available layer names/depths and exit",
    )
    return parser.parse_args()


def next_run_number(model_path, scope):
    prefix = f"bulk_random_projection_experiment_{scope}_"
    existing_runs = []
    if os.path.isdir(model_path):
        pattern = re.compile(re.escape(prefix) + r"(\d+)\.pt$")
        for filename in os.listdir(model_path):
            match = pattern.match(filename)
            if match:
                existing_runs.append(int(match.group(1)))
    return max(existing_runs) + 1 if existing_runs else 1


def resolve_target_layer(all_layers, scope):
    scope_value = scope
    try:
        scope_value = int(scope_value)
    except ValueError:
        pass

    if isinstance(scope_value, int):
        if scope_value < 0 or scope_value >= len(all_layers):
            raise ValueError(f"Depth index {scope_value} out of range [0, {len(all_layers) - 1}]")
        return scope_value, all_layers[scope_value]
    if scope_value == "first":
        return scope_value, all_layers[0]
    if scope_value == "last":
        return scope_value, all_layers[-1]

    matching = [entry for entry in all_layers if entry["name"] == scope_value]
    if not matching:
        available = [entry["name"] for entry in all_layers]
        raise ValueError(f"Unknown layer name '{scope_value}'. Available: {available}")
    return scope_value, matching[0]


def random_orthonormal_basis(n_params, seed, dtype=torch.float32):
    """Draw and randomly order a Haar-random orthonormal basis on the CPU."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    gaussian_matrix = torch.randn(n_params, n_params, generator=generator, dtype=dtype)
    basis, triangular = torch.linalg.qr(gaussian_matrix, mode="reduced")

    # Normalize the QR sign ambiguity to obtain the standard Gaussian-QR Haar draw.
    signs = torch.sign(torch.diagonal(triangular))
    signs[signs == 0] = 1
    basis = basis * signs.unsqueeze(0)

    # The columns of a Haar basis are exchangeable, but explicitly permuting them
    # records the requested random ordering independently of the QR implementation.
    order = torch.randperm(n_params, generator=generator)
    return basis[:, order].contiguous(), order


def measure_orthonormality(basis, max_full_dimension=4096, sample_columns=64):
    """Measure Q^T Q error without forming a huge Gram matrix."""
    n_params = basis.shape[1]
    if n_params <= max_full_dimension:
        checked_basis = basis
    else:
        indices = (
            torch.linspace(
                0,
                n_params - 1,
                steps=min(sample_columns, n_params),
            )
            .round()
            .long()
            .unique()
        )
        checked_basis = basis[:, indices]

    identity = torch.eye(
        checked_basis.shape[1],
        dtype=checked_basis.dtype,
    )
    error = torch.max(torch.abs(checked_basis.T @ checked_basis - identity)).item()
    return error, checked_basis.shape[1]


def initialize_payload(metadata, eval_loaders):
    payload = {
        "metadata": metadata,
        "on_projector": {},
        "out_projector": {},
    }
    for projector_name in ("on_projector", "out_projector"):
        for dataset_name in eval_loaders:
            payload[projector_name][dataset_name] = {
                "loss": {},
                "accuracy": {},
            }
    return payload


def main():
    args = parse_args()
    if args.num_k_points is not None and args.num_k_points <= 0:
        raise ValueError("--num-k-points must be positive")
    if args.num_calib_cifar < 0:
        raise ValueError("--num-calib-cifar must be non-negative")

    model_path = os.path.abspath(args.model_path)
    model = resnet20(num_classes=10)
    weights_path = os.path.join(model_path, "final_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))

    all_layers = _get_targeted_layers(model)
    if args.list_layers:
        print(f"{'Depth':<8} {'Name':<30} {'Type':<15} {'Params'}")
        print("-" * 70)
        for entry in all_layers:
            layer_type = type(entry["module"]).__name__
            print(
                f"{entry['depth']:<8} {entry['name']:<30} " f"{layer_type:<15} {entry['n_params']}"
            )
        return

    scope_value, target_layer = resolve_target_layer(all_layers, args.scope)
    depth = target_layer["depth"]
    n_params = target_layer["n_params"]
    if args.num_k_points is not None and args.num_k_points < n_params:
        k_values = np.unique(np.linspace(1, n_params, args.num_k_points, dtype=int))
    else:
        k_values = np.arange(1, n_params + 1)

    run_number = next_run_number(model_path, args.scope)
    seed = run_number if args.seed is None else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prefix = f"bulk_random_projection_experiment_{args.scope}_{run_number}"
    save_filename = os.path.join(model_path, f"{prefix}.pt")
    basis_filename = os.path.join(
        model_path,
        f"random_orthonormal_basis_{args.scope}_{run_number}.pt",
    )
    print(f"Run number: {run_number} (saving to {save_filename})")
    print(f"Random seed: {seed}")
    print(
        f"Targeting layer: depth={depth}, name='{target_layer['name']}', "
        f"n_params={n_params}, evaluating {len(k_values)} k-values from 1..{n_params}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cifar10_test_loader = DataLoader(
        datasets.CIFAR10(
            root=os.path.join(PROJECT_ROOT, "data"),
            train=False,
            download=True,
            transform=CIFAR10_TRANSFORM,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    cifar10_train_loader = DataLoader(
        datasets.CIFAR10(
            root=os.path.join(PROJECT_ROOT, "data"),
            train=True,
            download=True,
            transform=CIFAR10_TRANSFORM,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    gaussian_metadata = torch.load(
        os.path.join(model_path, "gaussian_metadata.pt"),
        map_location="cpu",
        weights_only=False,
    )
    gauss_labels = gaussian_metadata["labels"]
    gauss_images = gaussian_metadata["images"]
    randlabel_indices = gaussian_metadata["is_moment_matched"] == False
    matchlabel_indices = gaussian_metadata["is_moment_matched"] == True

    gauss_dataloader = DataLoader(
        TensorDataset(gauss_images, gauss_labels),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    gauss_dataloader_randlabels = DataLoader(
        TensorDataset(
            gauss_images[randlabel_indices],
            gauss_labels[randlabel_indices],
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    gauss_dataloader_matchlabels = DataLoader(
        TensorDataset(
            gauss_images[matchlabel_indices],
            gauss_labels[matchlabel_indices],
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    n_calib_cifar = min(args.num_calib_cifar, len(cifar10_train_loader.dataset))
    calib_cifar_indices = torch.randperm(len(cifar10_train_loader.dataset))[:n_calib_cifar]
    calib_cifar_subset = torch.utils.data.Subset(cifar10_train_loader.dataset, calib_cifar_indices)
    calibration_dataset = ConcatDataset(
        [
            calib_cifar_subset,
            TensorDatasetWithIntLabels(gauss_images, gauss_labels),
        ]
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    print(
        f"BN calibration set: {n_calib_cifar} CIFAR + "
        f"{len(gauss_images)} Gaussian = {len(calibration_dataset)} samples "
        f"({len(calibration_loader)} batches)"
    )

    t_basis = time.time()
    basis_cpu, basis_order = random_orthonormal_basis(
        n_params,
        seed,
        dtype=target_layer["module"].weight.dtype,
    )
    orthonormality_error, orthonormality_check_columns = measure_orthonormality(basis_cpu)
    print(
        f"Random basis generation took {time.time() - t_basis:.1f}s; "
        f"max checked |Q^T Q - I|={orthonormality_error:.3e} "
        f"({orthonormality_check_columns}/{n_params} columns)"
    )

    saved_basis_filename = None
    if not args.no_save_basis:
        basis_payload = {
            "basis": basis_cpu,
            "basis_order": basis_order,
            "seed": seed,
            "scope": args.scope,
            "layer_depth": depth,
            "layer_name": target_layer["name"],
            "n_params": n_params,
            "orthonormality_error": orthonormality_error,
            "orthonormality_check_columns": orthonormality_check_columns,
        }
        save_experiment_data(basis_payload, basis_filename)
        saved_basis_filename = os.path.basename(basis_filename)
        print(f"Saved random orthonormal basis to {basis_filename}")
    else:
        print("Skipping dense basis save; seed and permutation remain in results")

    basis_gpu = basis_cpu.to(
        device=device,
        dtype=target_layer["module"].weight.dtype,
    )
    original_weight = target_layer["module"].weight.data.clone()
    w_flat_gpu = original_weight.reshape(-1).to(device)
    all_coeffs = basis_gpu.T @ w_flat_gpu

    working_model = copy.deepcopy(model).to(device)
    working_layers = _get_targeted_layers(working_model)
    working_by_depth = {entry["depth"]: entry for entry in working_layers}
    working_layer_module = working_by_depth[depth]["module"]

    eval_loaders = {
        "test": cifar10_test_loader,
        "train": cifar10_train_loader,
        "gauss": gauss_dataloader,
        "gauss_randlabels": gauss_dataloader_randlabels,
        "gauss_matchlabels": gauss_dataloader_matchlabels,
    }
    metadata = {
        "experiment": "bulk_random_projection_control",
        "model_path": model_path,
        "scope": args.scope,
        "scope_value": scope_value,
        "layer_depth": depth,
        "layer_name": target_layer["name"],
        "n_params": n_params,
        "k_values": [int(k) for k in k_values],
        "seed": seed,
        "run_number": run_number,
        "basis_file": saved_basis_filename,
        "basis_order": basis_order.tolist(),
        "orthonormality_error": orthonormality_error,
        "orthonormality_check_columns": orthonormality_check_columns,
        "num_calib_cifar": n_calib_cifar,
    }
    data_payload = initialize_payload(metadata, eval_loaders)

    print(f"Projecting model onto nested prefixes of the random basis for " f"scope={args.scope}")
    t_loop = time.time()
    for i, k in enumerate(k_values):
        k_int = int(k)
        selected_vectors = basis_gpu[:, :k_int]
        selected_coeffs = all_coeffs[:k_int]

        w_on = selected_vectors @ selected_coeffs
        working_layer_module.weight.data.copy_(w_on.view_as(original_weight))
        _recalibrate_batchnorm(working_model, calibration_loader, device)
        for dataset_name, loader in eval_loaders.items():
            evaluation = evaluate(working_model, loader, device)
            data_payload["on_projector"][dataset_name]["loss"][k_int] = evaluation["loss"]
            data_payload["on_projector"][dataset_name]["accuracy"][k_int] = evaluation["accuracy"]

        w_out = w_flat_gpu - w_on
        working_layer_module.weight.data.copy_(w_out.view_as(original_weight))
        _recalibrate_batchnorm(working_model, calibration_loader, device)
        for dataset_name, loader in eval_loaders.items():
            evaluation = evaluate(working_model, loader, device)
            data_payload["out_projector"][dataset_name]["loss"][k_int] = evaluation["loss"]
            data_payload["out_projector"][dataset_name]["accuracy"][k_int] = evaluation["accuracy"]

        working_layer_module.weight.data.copy_(original_weight)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_loop
            per_iteration = elapsed / (i + 1)
            eta = per_iteration * (len(k_values) - i - 1)
            print(
                f"Completed {i + 1}/{len(k_values)} (k={k_int}) | "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
            )
            save_experiment_data(data_payload, save_filename)

    save_experiment_data(data_payload, save_filename)
    print(f"Done. Total loop time: {time.time() - t_loop:.0f}s")


if __name__ == "__main__":
    main()
