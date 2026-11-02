"""
Utility functions for model evaluation, checkpointing, and experiment I/O.

Provides a consistent interface for:
    - Evaluating model accuracy and loss on arbitrary DataLoaders
    - Saving training checkpoints and loading model weights
    - Persisting experiment artifacts (metrics dicts, numpy arrays, tensors)
    - Auto-generating unique, descriptive experiment directory names

Naming convention for experiment directories:
    <base_dir>/<experiment_name>_run<NNN>/
    where experiment_name encodes arch, dataset, gaussian config, and seed.
    The run number auto-increments to avoid overwriting previous runs.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Experiment naming


def generate_experiment_name(arch, dataset, gaussian_count=0, random_labels=True, seed=None):
    """Build a descriptive experiment name string from configuration.

    Examples:
        resnet20_cifar10gaussian_g500_randlabels
        resnet20_cifar10gaussian_g500_randlabels_s42
        resnet20_cifar10
        resnet32_gaussian_g1000_matched_s7
    """
    parts = [arch]

    dataset_tag = dataset.replace("_", "")
    parts.append(dataset_tag)

    if dataset != "cifar10":
        parts.append(f"g{gaussian_count}")
        parts.append("randlabels" if random_labels else "matched")

    if seed is not None:
        parts.append(f"s{seed}")

    return "_".join(parts)


def resolve_experiment_dir(base_dir, experiment_name, run_name=None):
    """Find the next available run directory for an experiment.

    If run_name is provided, uses it directly:
        <base_dir>/<run_name>/

    Otherwise, auto-increments:
        <base_dir>/<experiment_name>_run001/
        <base_dir>/<experiment_name>_run002/
        ...
    """
    base_dir = Path(base_dir)

    if run_name is not None:
        return base_dir / run_name

    pattern = re.compile(re.escape(experiment_name) + r"_run(\d+)$")
    max_run = 0

    if base_dir.exists():
        for entry in base_dir.iterdir():
            if entry.is_dir():
                match = pattern.match(entry.name)
                if match:
                    max_run = max(max_run, int(match.group(1)))

    next_run = max_run + 1
    return base_dir / f"{experiment_name}_run{next_run:03d}"


# Evaluation


def evaluate(model, dataloader, device):
    """Compute cross-entropy loss and accuracy on a DataLoader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            predictions = logits.argmax(dim=1)
            total_correct += predictions.eq(labels).sum().item()
            total_samples += labels.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


# Checkpointing and model loading


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """Save a training checkpoint."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, filepath)


def strip_data_parallel_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys added by DataParallel."""
    return {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def load_model_weights(filepath, model, device=None):
    """Load model weights, handling DataParallel prefix mismatches.

    Tries loading directly first. If that fails due to key mismatches
    (e.g., saved with DataParallel, loading without), strips the
    'module.' prefix and retries.
    """
    map_location = device if device is not None else None
    raw = torch.load(filepath, map_location=map_location, weights_only=False)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    else:
        state_dict = raw

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(strip_data_parallel_prefix(state_dict))

    return model


# Experiment data I/O


def save_experiment_data(data, filepath):
    """Save experiment data, choosing format based on file extension.

    Supported formats:
        .pt / .pth  -- torch.save (dicts, tensors, any picklable object)
        .npy        -- numpy.save (single numpy array)
        .npz        -- numpy.savez (dict of numpy arrays)
        .json       -- json.dump (JSON-serializable dicts/lists)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    suffix = filepath.suffix.lower()

    if suffix in (".pt", ".pth"):
        torch.save(data, filepath)
    elif suffix == ".npy":
        np.save(filepath, data)
    elif suffix == ".npz":
        np.savez(filepath, **data)
    elif suffix == ".json":
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. " "Use .pt, .pth, .npy, .npz, or .json."
        )
