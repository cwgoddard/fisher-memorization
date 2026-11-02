"""Train ResNet-20 on CIFAR-10 with synthetic label noise."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import _bootstrap  # noqa: F401
from src.label_noise_data import apply_label_noise, load_cifar10
from src.resnet import resnet20

EPOCHS = 200
BATCH_SIZE = 128


def cache_to_gpu(dataset, device, bs=1000):
    """Cache deterministic dataset transforms as device tensors."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True
    )
    xs, ys = zip(*[(x, y) for x, y in loader])
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def evaluate(X, Y, model, bs=1000):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, X.size(0), bs):
            output = model(X[i : i + bs])
            targets = Y[i : i + bs]
            total_loss += F.cross_entropy(output, targets, reduction="sum").item()
            correct += output.argmax(1).eq(targets).sum().item()
    return total_loss / X.size(0), correct / X.size(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise-fraction", type=float, default=0.10, help="Fraction of training labels to corrupt"
    )
    parser.add_argument(
        "--noise-seed", type=int, default=42, help="Seeds the label-corruption draw"
    )
    parser.add_argument("--seed", type=int, default=None, help="Seeds weight init and batch order")
    parser.add_argument("--save-dir", type=str, default="output/noisy10pct")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.seed is not None:
        # cudnn.benchmark stays on, so runs are reproducible up to
        # kernel-selection nondeterminism: enough for independent replicates,
        # not for bitwise equality.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    train_ds, test_ds = load_cifar10(args.data_root)

    meta = None
    if args.noise_fraction > 0:
        meta = apply_label_noise(train_ds, args.noise_fraction, seed=args.noise_seed)
        meta["init_seed"] = -1 if args.seed is None else args.seed
        n_noised = meta["is_noised"].sum()
        print(
            f"Corrupted {n_noised}/{len(train_ds)} labels "
            f"({100 * n_noised / len(train_ds):.2f}%)"
        )

    # Cached after the noise is applied, so the corrupted targets are stored.
    t0 = time.time()
    X_train, Y_train = cache_to_gpu(train_ds, args.device)
    X_test, Y_test = cache_to_gpu(test_ds, args.device)
    print(
        f"Cached {X_train.size(0):,} train + {X_test.size(0):,} test images "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )

    model = resnet20().to(args.device)
    cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    N = X_train.size(0)
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        # Same semantics as DataLoader(shuffle=True, drop_last=False).
        perm = torch.randperm(N, device=X_train.device)
        for i in range(0, N, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            loss = F.cross_entropy(model(X_train[idx]), Y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            tr_loss, tr_acc = evaluate(X_train, Y_train, model)
            te_loss, te_acc = evaluate(X_test, Y_test, model)
            print(
                f"Epoch {epoch:3d}  train {tr_loss:.4f}/{tr_acc:.4f}  "
                f"test {te_loss:.4f}/{te_acc:.4f}  "
                f"[{time.time() - t0:.0f}s]",
                flush=True,
            )

    torch.save(model.state_dict(), save_dir / "model.pt")
    if meta is not None:
        np.savez(save_dir / "noise_metadata.npz", **meta)
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()
