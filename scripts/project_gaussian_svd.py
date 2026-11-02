"""Layerwise SVD control for the CIFAR-10 + Gaussian input-noise experiment."""

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
        description=(
            "Bulk layerwise rank-r SVD projection control for the "
            "CIFAR-10 + Gaussian input-noise experiment."
        )
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
            "Seed for calibration sampling. By default, use the "
            "auto-incremented experiment run number."
        ),
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print available layer names/depths and matrix shapes, then exit",
    )
    return parser.parse_args()


def next_run_number(model_path, scope):
    prefix = f"bulk_svd_projection_experiment_{scope}_"
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


def matrix_shape(weight):
    """Return the matrix shape used for SVD of a Conv2d/Linear weight."""
    rows = weight.shape[0]
    columns = weight.numel() // rows
    return rows, columns


def svd_axis_values(rows, columns):
    """Return axis conventions A, B, and C for every attainable rank."""
    max_rank = min(rows, columns)
    ranks = np.arange(1, max_rank + 1, dtype=int)
    subspace_dimensions = ranks * columns
    manifold_degrees = ranks * (rows + columns - ranks)
    return ranks, subspace_dimensions, manifold_degrees


def initialize_payload(metadata, eval_loaders):
    payload = {
        "metadata": metadata,
        "k_rank": metadata["k_rank"],
        "k_values": metadata["k_values"],
        "k_manifold_dof": metadata["k_manifold_dof"],
        "singular_values": metadata["singular_values"],
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
    if args.num_calib_cifar < 0:
        raise ValueError("--num-calib-cifar must be non-negative")

    model_path = os.path.abspath(args.model_path)
    model = resnet20(num_classes=10)
    weights_path = os.path.join(model_path, "final_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))

    all_layers = _get_targeted_layers(model)
    if args.list_layers:
        print(f"{'Depth':<8} {'Name':<30} {'Type':<12} " f"{'Matrix shape':<16} {'Max rank'}")
        print("-" * 85)
        for entry in all_layers:
            rows, columns = matrix_shape(entry["module"].weight)
            shape = f"{rows} x {columns}"
            print(
                f"{entry['depth']:<8} {entry['name']:<30} "
                f"{type(entry['module']).__name__:<12} "
                f"{shape:<16} {min(rows, columns)}"
            )
        return

    scope_value, target_layer = resolve_target_layer(all_layers, args.scope)
    depth = target_layer["depth"]
    original_weight_cpu = target_layer["module"].weight.detach().cpu().clone()
    rows, columns = matrix_shape(original_weight_cpu)
    weight_matrix_cpu = original_weight_cpu.reshape(rows, columns)
    ranks, k_values, manifold_degrees = svd_axis_values(rows, columns)

    run_number = next_run_number(model_path, args.scope)
    seed = run_number if args.seed is None else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prefix = f"bulk_svd_projection_experiment_{args.scope}_{run_number}"
    save_filename = os.path.join(model_path, f"{prefix}.pt")
    print(f"Run number: {run_number} (saving to {save_filename})")
    print(f"Calibration seed: {seed}")
    print(
        f"Targeting layer: depth={depth}, name='{target_layer['name']}', "
        f"weight shape={tuple(original_weight_cpu.shape)}, "
        f"SVD matrix={rows}x{columns}, ranks=1..{len(ranks)}"
    )
    print(
        f"Primary axis convention: k=r*n, ranging from "
        f"{int(k_values[0])} to {int(k_values[-1])}"
    )

    t_svd = time.time()
    u_cpu, singular_values_cpu, vh_cpu = torch.linalg.svd(
        weight_matrix_cpu,
        full_matrices=False,
    )
    full_reconstruction = (u_cpu * singular_values_cpu.unsqueeze(0)) @ vh_cpu
    reconstruction_error = torch.max(torch.abs(full_reconstruction - weight_matrix_cpu)).item()
    print(
        f"Compact SVD took {time.time() - t_svd:.3f}s; "
        f"max full-rank reconstruction error={reconstruction_error:.3e}"
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
    moment_matched = gaussian_metadata["is_moment_matched"].bool()

    gauss_dataloader = DataLoader(
        TensorDataset(gauss_images, gauss_labels),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    gauss_dataloader_randlabels = DataLoader(
        TensorDataset(
            gauss_images[~moment_matched],
            gauss_labels[~moment_matched],
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    gauss_dataloader_matchlabels = DataLoader(
        TensorDataset(
            gauss_images[moment_matched],
            gauss_labels[moment_matched],
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

    working_model = copy.deepcopy(model).to(device)
    working_layers = _get_targeted_layers(working_model)
    working_by_depth = {entry["depth"]: entry for entry in working_layers}
    working_layer_module = working_by_depth[depth]["module"]
    original_weight = working_layer_module.weight.data.clone()

    factor_dtype = original_weight.dtype
    u = u_cpu.to(device=device, dtype=factor_dtype)
    singular_values = singular_values_cpu.to(
        device=device,
        dtype=factor_dtype,
    )
    vh = vh_cpu.to(device=device, dtype=factor_dtype)

    eval_loaders = {
        "test": cifar10_test_loader,
        "train": cifar10_train_loader,
        "gauss": gauss_dataloader,
        "gauss_randlabels": gauss_dataloader_randlabels,
        "gauss_matchlabels": gauss_dataloader_matchlabels,
    }
    metadata = {
        "experiment": "bulk_svd_projection_control",
        "model_path": model_path,
        "scope": args.scope,
        "scope_value": scope_value,
        "layer_depth": depth,
        "layer_name": target_layer["name"],
        "weight_shape": list(original_weight.shape),
        "matrix_shape": [rows, columns],
        "n_params": original_weight.numel(),
        "axis_convention": "B: linear subspace dimension k = rank * n_columns",
        "k_rank": [int(rank) for rank in ranks],
        "k_values": [int(k) for k in k_values],
        "k_manifold_dof": [int(k) for k in manifold_degrees],
        "singular_values": singular_values_cpu.tolist(),
        "full_rank_reconstruction_error": reconstruction_error,
        "seed": seed,
        "run_number": run_number,
        "num_calib_cifar": n_calib_cifar,
    }
    data_payload = initialize_payload(metadata, eval_loaders)

    print("Evaluating rank-r SVD truncations and complementary residuals")
    t_loop = time.time()
    for i, (rank, k_value) in enumerate(zip(ranks, k_values)):
        rank_int = int(rank)
        k_int = int(k_value)

        w_on_matrix = (u[:, :rank_int] * singular_values[:rank_int].unsqueeze(0)) @ vh[:rank_int, :]
        working_layer_module.weight.data.copy_(w_on_matrix.reshape_as(original_weight))
        _recalibrate_batchnorm(working_model, calibration_loader, device)
        for dataset_name, loader in eval_loaders.items():
            evaluation = evaluate(working_model, loader, device)
            data_payload["on_projector"][dataset_name]["loss"][k_int] = evaluation["loss"]
            data_payload["on_projector"][dataset_name]["accuracy"][k_int] = evaluation["accuracy"]

        w_out_matrix = original_weight.reshape(rows, columns) - w_on_matrix
        working_layer_module.weight.data.copy_(w_out_matrix.reshape_as(original_weight))
        _recalibrate_batchnorm(working_model, calibration_loader, device)
        for dataset_name, loader in eval_loaders.items():
            evaluation = evaluate(working_model, loader, device)
            data_payload["out_projector"][dataset_name]["loss"][k_int] = evaluation["loss"]
            data_payload["out_projector"][dataset_name]["accuracy"][k_int] = evaluation["accuracy"]

        working_layer_module.weight.data.copy_(original_weight)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_loop
            per_iteration = elapsed / (i + 1)
            eta = per_iteration * (len(ranks) - i - 1)
            print(
                f"Completed {i + 1}/{len(ranks)} "
                f"(rank={rank_int}, k={k_int}) | "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
            )
            save_experiment_data(data_payload, save_filename)

    save_experiment_data(data_payload, save_filename)
    print(f"Done. Total loop time: {time.time() - t_loop:.0f}s")


if __name__ == "__main__":
    main()
