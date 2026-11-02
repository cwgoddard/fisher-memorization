"""Training script for ResNet on CIFAR-10 with optional Gaussian noise samples."""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data

from _bootstrap import PROJECT_ROOT
from src.experiment_utils import (
    evaluate,
    generate_experiment_name,
    resolve_experiment_dir,
    save_checkpoint,
    save_experiment_data,
    strip_data_parallel_prefix,
)
from src.gaussian_data import (
    TensorDatasetWithIntLabels,
    build_combined_dataset,
    compute_class_statistics,
    generate_moment_matched_gaussian,
    load_cifar10,
)
from src.gaussian_resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202

ARCH_REGISTRY = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet1202": resnet1202,
}


def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train ResNet on CIFAR-10 with optional Gaussian noise samples"
    )

    parser.add_argument(
        "--arch",
        default="resnet20",
        choices=list(ARCH_REGISTRY.keys()),
        help="Model architecture (default: resnet20)",
    )
    parser.add_argument(
        "--dataset",
        default="cifar10_gaussian",
        choices=["cifar10", "gaussian", "cifar10_gaussian"],
        help="Dataset mode (default: cifar10_gaussian)",
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(PROJECT_ROOT, "data"),
        help="Path to CIFAR-10 data directory",
    )
    parser.add_argument(
        "--gaussian-count",
        default=10000,
        type=int,
        help="Number of Gaussian noise samples to include (default: 10000)",
    )
    parser.add_argument(
        "--matched-labels",
        action="store_true",
        help="Use moment-matched labels (label = source class) instead of random",
    )
    parser.add_argument("--epochs", default=200, type=int, help="Total training epochs")
    parser.add_argument("--batch-size", default=128, type=int, help="Training batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay")
    parser.add_argument(
        "--lr-milestones",
        nargs="+",
        default=[100, 150],
        type=int,
        help="Epochs at which to decay learning rate by 10x (default: 100 150)",
    )
    parser.add_argument("--workers", default=4, type=int, help="DataLoader worker count")
    parser.add_argument("--log-freq", default=50, type=int, help="Batch log frequency")
    parser.add_argument(
        "--save-dir",
        default=os.path.join(PROJECT_ROOT, "experiments"),
        help="Base directory for saving outputs (default: experiments)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        type=str,
        help="Explicit run name (overrides auto-generated name)",
    )
    parser.add_argument("--seed", default=None, type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--no-data-parallel",
        action="store_true",
        help="Disable nn.DataParallel wrapping",
    )

    return parser.parse_args()


def build_datasets(args):
    """Construct train and test datasets according to the chosen mode.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dict with keys:
            "train_dataset", "test_dataset", "description",
            and optionally "gaussian_metadata" and "cifar_train".
    """
    random_labels = not args.matched_labels

    if args.dataset == "cifar10":
        cifar_train, cifar_test = load_cifar10(args.data_root)
        return {
            "train_dataset": cifar_train,
            "test_dataset": cifar_test,
            "cifar_train": cifar_train,
            "description": "CIFAR-10",
        }

    elif args.dataset == "gaussian":
        cifar_train, cifar_test = load_cifar10(args.data_root)
        class_means, class_stds = compute_class_statistics(cifar_train)
        gaussian_data = generate_moment_matched_gaussian(
            class_means,
            class_stds,
            num_samples=args.gaussian_count,
            random_labels=random_labels,
        )
        gaussian_dataset = TensorDatasetWithIntLabels(
            gaussian_data["images"], gaussian_data["labels"]
        )
        label_type = "random" if random_labels else "matched"
        return {
            "train_dataset": gaussian_dataset,
            "test_dataset": cifar_test,
            "description": f"{args.gaussian_count} Gaussian samples ({label_type} labels)",
            "gaussian_metadata": gaussian_data,
        }

    elif args.dataset == "cifar10_gaussian":
        result = build_combined_dataset(
            data_root=args.data_root,
            gaussian_count=args.gaussian_count,
            random_labels=random_labels,
        )
        label_type = "random" if random_labels else "matched"
        return {
            "train_dataset": result["train_dataset"],
            "test_dataset": result["test_dataset"],
            "cifar_train": result["cifar_train"],
            "description": (
                f"CIFAR-10 + {args.gaussian_count} Gaussian samples " f"({label_type} labels)"
            ),
            "gaussian_metadata": result["gaussian_metadata"],
        }

    raise ValueError(f"Unknown dataset mode: {args.dataset}")


def train_one_epoch(model, dataloader, criterion, optimizer, epoch, log_freq, device):
    """Run one training epoch.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        epoch: Current epoch index.
        log_freq: Print every log_freq batches.
        device: torch.device for computation.

    Returns:
        Dict with "loss" (average training loss for the epoch).
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if batch_idx % log_freq == 0:
            print(f"  Epoch [{epoch}][{batch_idx}/{len(dataloader)}]  " f"Loss: {loss.item():.4f}")

    return {"loss": running_loss / max(num_batches, 1)}


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random_labels = not args.matched_labels
    experiment_name = generate_experiment_name(
        arch=args.arch,
        dataset=args.dataset,
        gaussian_count=args.gaussian_count,
        random_labels=random_labels,
        seed=args.seed,
    )
    save_path = resolve_experiment_dir(
        base_dir=args.save_dir,
        experiment_name=experiment_name,
        run_name=args.run_name,
    )
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Experiment:    {experiment_name}")
    print(f"Architecture:  {args.arch}")
    print(f"Dataset mode:  {args.dataset}")
    print(f"Save directory: {save_path}")
    print(f"Device:        {device}")
    print()

    print("Loading datasets...")
    dataset_bundle = build_datasets(args)
    train_dataset = dataset_bundle["train_dataset"]
    test_dataset = dataset_bundle["test_dataset"]
    print(f"  {dataset_bundle['description']}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    if "gaussian_metadata" in dataset_bundle:
        metadata = dataset_bundle["gaussian_metadata"]
        torch.save(metadata, save_path / "gaussian_metadata.pt")
        n_matched = metadata["is_moment_matched"].sum().item()
        print(
            f"  Gaussian subset: {len(metadata['labels'])} samples, " f"{n_matched} moment-matched"
        )

    config = {
        "arch": args.arch,
        "dataset": args.dataset,
        "data_root": str(Path(args.data_root).resolve()),
        "description": dataset_bundle["description"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_milestones": args.lr_milestones,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "gaussian_count": args.gaussian_count,
        "random_labels": random_labels,
        "seed": args.seed,
        "experiment_name": experiment_name,
        "save_path": str(save_path),
    }
    save_experiment_data(config, save_path / "config.json")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(f"\nBuilding model: {args.arch}")
    model = ARCH_REGISTRY[args.arch]()
    if not args.no_data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
    )

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    print(f"\nTraining for {args.epochs} epochs\n")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs}  lr={current_lr:.5e}")

        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
            args.log_freq,
            device,
        )
        scheduler.step()

        train_eval = evaluate(model, train_loader, device)
        test_eval = evaluate(model, test_loader, device)

        history["train_loss"].append(train_eval["loss"])
        history["train_acc"].append(train_eval["accuracy"])
        history["test_loss"].append(test_eval["loss"])
        history["test_acc"].append(test_eval["accuracy"])

        elapsed = time.time() - epoch_start
        print(f"  Train -- loss: {train_eval['loss']:.4f}, " f"acc: {train_eval['accuracy']:.4f}")
        print(f"  Test  -- loss: {test_eval['loss']:.4f}, " f"acc: {test_eval['accuracy']:.4f}")
        print(f"  Time: {elapsed:.1f}s\n")

        if (epoch + 1) % args.log_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                {"train": train_eval, "test": test_eval},
                save_path / "checkpoints" / f"epoch_{epoch:04d}.pt",
            )

    final_state = strip_data_parallel_prefix(model.state_dict())
    torch.save(final_state, save_path / "final_model.pt")
    save_experiment_data(history, save_path / "training_history.pt")

    print("Training complete.")
    print(f"  Final train acc: {history['train_acc'][-1]:.4f}")
    print(f"  Final test acc:  {history['test_acc'][-1]:.4f}")
    print(f"  Outputs saved to: {save_path}")


if __name__ == "__main__":
    main()
