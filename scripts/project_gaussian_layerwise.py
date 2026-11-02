"""Per-layer Fisher projection sweeps for the input-noise models (Figs. 1, 7 and 11)."""

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
from src.fisher_projection import (
    LayerwiseFisherProjector,
    _get_targeted_layers,
    _recalibrate_batchnorm,
)
from src.gaussian_data import CIFAR10_TRANSFORM, TensorDatasetWithIntLabels
from src.gaussian_resnet import resnet20

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-path",
    type=str,
    default=os.path.join(
        PROJECT_ROOT, "experiments/resnet20_cifar10gaussian_g500_randlabels_run004"
    ),
)
parser.add_argument(
    "--scope",
    type=str,
    default="first",
    help="Layer to target: 'first', 'last', an integer depth index, or a layer name (use --list-layers to see options)",
)
parser.add_argument(
    "--num-k-points",
    type=int,
    default=None,
    help="Cap on number of k-values to evaluate (uniformly spaced). Default: all.",
)
parser.add_argument(
    "--num-calib-cifar",
    type=int,
    default=4500,
    help="Number of CIFAR train samples in BN calibration set (all Gaussian images are always included)",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=256,
    help="Batch size for Fisher, BN calibration, and evaluation",
)
parser.add_argument(
    "--output-dir", type=str, default=None, help="Result directory (default: --model-path)"
)
parser.add_argument(
    "--output-tag",
    type=str,
    default="",
    help="Optional tag inserted into result and Fisher filenames",
)
parser.add_argument(
    "--bn-averaging",
    choices=("exponential", "cumulative"),
    default="exponential",
    help="BatchNorm running-stat averaging during recalibration",
)
parser.add_argument(
    "--include-k-zero", action="store_true", help="Include k=0 as a matched BN-only control"
)
parser.add_argument(
    "--fail-if-output-exists",
    action="store_true",
    help="Refuse to reuse an output namespace containing results",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--list-layers", action="store_true", help="Print available layer names/depths and exit"
)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

output_dir = args.output_dir or args.model_path
os.makedirs(output_dir, exist_ok=True)
tag_prefix = f"{args.output_tag}_" if args.output_tag else ""
prefix = f"bulk_fisher_experiment_{tag_prefix}{args.scope}_"
existing_runs = []
if os.path.isdir(output_dir):
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.pt$")
    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if m:
            existing_runs.append(int(m.group(1)))
if args.fail_if_output_exists and existing_runs:
    raise FileExistsError(
        f"Refusing to reuse populated output namespace: {output_dir}/{prefix}*.pt"
    )
run_number = max(existing_runs) + 1 if existing_runs else 1
save_filename = os.path.join(output_dir, f"{prefix}{run_number}.pt")
print(f"Run number: {run_number} (saving to {save_filename})")

path = os.path.join(args.model_path, "final_model.pt")
model = resnet20(num_classes=10)
model.load_state_dict(torch.load(path))

all_layers = _get_targeted_layers(model)

if args.list_layers:
    print(f"{'Depth':<8} {'Name':<30} {'Type':<15} {'Params'}")
    print("-" * 70)
    for entry in all_layers:
        layer_type = type(entry["module"]).__name__
        print(f"{entry['depth']:<8} {entry['name']:<30} {layer_type:<15} {entry['n_params']}")
    exit(0)

scope_value = args.scope
try:
    scope_value = int(scope_value)
except ValueError:
    pass

if isinstance(scope_value, int):
    if scope_value < 0 or scope_value >= len(all_layers):
        raise ValueError(f"Depth index {scope_value} out of range [0, {len(all_layers)-1}]")
    target_layer = all_layers[scope_value]
elif scope_value == "first":
    target_layer = all_layers[0]
elif scope_value == "last":
    target_layer = all_layers[-1]
else:
    matching = [e for e in all_layers if e["name"] == scope_value]
    if not matching:
        available = [e["name"] for e in all_layers]
        raise ValueError(f"Unknown layer name '{scope_value}'. Available: {available}")
    target_layer = matching[0]

n_params = target_layer["n_params"]
if args.num_k_points is not None and args.num_k_points < n_params:
    k_values = np.unique(np.linspace(1, n_params, args.num_k_points, dtype=int))
else:
    k_values = np.arange(1, n_params + 1)
if args.include_k_zero:
    k_values = np.concatenate(([0], k_values))

print(
    f"Targeting layer: depth={target_layer['depth']}, name='{target_layer['name']}', "
    f"n_params={n_params}, evaluating {len(k_values)} k-values "
    f"from {int(k_values[0])}..{n_params}"
)

NUM_WORKERS = 0
BATCH_SIZE = args.batch_size

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

gaussian_metadata_path = os.path.join(args.model_path, "gaussian_metadata.pt")
gaussian_metadata = torch.load(gaussian_metadata_path)
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
    TensorDataset(gauss_images[randlabel_indices], gauss_labels[randlabel_indices]),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)
gauss_dataloader_matchlabels = DataLoader(
    TensorDataset(gauss_images[matchlabel_indices], gauss_labels[matchlabel_indices]),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


cifar_plus_gauss_dataset = ConcatDataset(
    [cifar10_train_loader.dataset, TensorDatasetWithIntLabels(gauss_images, gauss_labels)]
)

cifar_plus_gauss_dataloader = DataLoader(
    cifar_plus_gauss_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

calib_cifar_indices = torch.randperm(len(cifar10_train_loader.dataset))[: args.num_calib_cifar]
calib_cifar_subset = torch.utils.data.Subset(cifar10_train_loader.dataset, calib_cifar_indices)
calibration_dataset = ConcatDataset(
    [calib_cifar_subset, TensorDatasetWithIntLabels(gauss_images, gauss_labels)]
)
calibration_loader = DataLoader(
    calibration_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
print(
    f"BN calibration set: {args.num_calib_cifar} CIFAR + {len(gauss_images)} Gaussian "
    f"= {len(calibration_dataset)} samples ({len(calibration_loader)} batches)"
)

t0 = time.time()
projector = LayerwiseFisherProjector(
    model,
    cifar_plus_gauss_dataloader,
    device,
    layer_scope=scope_value,
    fisher_batch_size=BATCH_SIZE,
    fisher_num_workers=NUM_WORKERS,
)
print(f"Fisher computation + eigendecomposition took {time.time() - t0:.1f}s")

depth = list(projector.selected_depths)[0]

fisher_tag = f"_{args.output_tag}" if args.output_tag else ""
fisher_save_path = os.path.join(output_dir, f"fisher_matrix{fisher_tag}_{depth}.pt")
if args.fail_if_output_exists and os.path.exists(fisher_save_path):
    raise FileExistsError(f"Refusing to overwrite existing Fisher matrix: {fisher_save_path}")
torch.save(projector.fisher_by_depth[depth], fisher_save_path)
print(f"Saved Fisher matrix to {fisher_save_path}")
_, evecs_cached = projector.eigen_by_depth[depth]
evecs_gpu = evecs_cached.to(device=device, dtype=target_layer["module"].weight.dtype)

original_weight = target_layer["module"].weight.data.clone()
w_flat_gpu = original_weight.reshape(-1).to(device)
all_coeffs = evecs_gpu.T @ w_flat_gpu

working_model = copy.deepcopy(model).to(device)
working_layers = _get_targeted_layers(working_model)
working_by_depth = {e["depth"]: e for e in working_layers}
working_layer_module = working_by_depth[depth]["module"]

print(f"Projecting model onto top-k Fisher eigenvectors for scope={args.scope}")

eval_loaders = {
    "test": cifar10_test_loader,
    "train": cifar10_train_loader,
    "gauss": gauss_dataloader,
    "gauss_randlabels": gauss_dataloader_randlabels,
    "gauss_matchlabels": gauss_dataloader_matchlabels,
}

data_payload = {
    "metadata": {
        "model_path": args.model_path,
        "scope": args.scope,
        "layer_depth": depth,
        "layer_name": target_layer["name"],
        "n_params": n_params,
        "num_calib_cifar": args.num_calib_cifar,
        "num_calib_gaussian": len(gauss_images),
        "batch_size": args.batch_size,
        "bn_averaging": args.bn_averaging,
        "include_k_zero": args.include_k_zero,
        "seed": args.seed,
        "output_tag": args.output_tag,
    },
    "on_projector": {},
    "out_projector": {},
}
for proj in ("on_projector", "out_projector"):
    for name in eval_loaders:
        data_payload[proj][name] = {"loss": {}, "accuracy": {}}

t_loop = time.time()
for i, k in enumerate(k_values):
    k_int = int(k)
    if k_int == 0:
        w_on = torch.zeros_like(w_flat_gpu)
    else:
        k_eff = min(k_int, evecs_gpu.shape[1])
        top_evecs = evecs_gpu[:, -k_eff:]
        top_coeffs = all_coeffs[-k_eff:]
        w_on = top_evecs @ top_coeffs
    working_layer_module.weight.data.copy_(w_on.view_as(original_weight))
    _recalibrate_batchnorm(
        working_model, calibration_loader, device, cumulative=args.bn_averaging == "cumulative"
    )
    for dataset_name, loader in eval_loaders.items():
        evaluation = evaluate(working_model, loader, device)
        data_payload["on_projector"][dataset_name]["loss"][k_int] = evaluation["loss"]
        data_payload["on_projector"][dataset_name]["accuracy"][k_int] = evaluation["accuracy"]

    w_out = w_flat_gpu - w_on
    working_layer_module.weight.data.copy_(w_out.view_as(original_weight))
    _recalibrate_batchnorm(
        working_model, calibration_loader, device, cumulative=args.bn_averaging == "cumulative"
    )
    for dataset_name, loader in eval_loaders.items():
        evaluation = evaluate(working_model, loader, device)
        data_payload["out_projector"][dataset_name]["loss"][k_int] = evaluation["loss"]
        data_payload["out_projector"][dataset_name]["accuracy"][k_int] = evaluation["accuracy"]

    working_layer_module.weight.data.copy_(original_weight)

    if (i + 1) % 10 == 0:
        elapsed = time.time() - t_loop
        per_iter = elapsed / (i + 1)
        eta = per_iter * (len(k_values) - i - 1)
        print(
            f"Completed {i + 1}/{len(k_values)} (k={k_int}) | "
            f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
        )
        save_experiment_data(data_payload, save_filename)

save_experiment_data(data_payload, save_filename)
print(f"Done. Total loop time: {time.time() - t_loop:.0f}s")
