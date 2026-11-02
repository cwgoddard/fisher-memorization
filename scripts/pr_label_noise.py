"""Layerwise Fisher participation-ratio analysis for label-noise CIFAR-10 models."""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from _bootstrap import PROJECT_ROOT
from src.experiment_utils import load_model_weights, save_experiment_data
from src.fisher_projection import (
    _eigh_symmetric,
    _get_targeted_layers,
    layerwise_fisher_wrt_weights,
)
from src.gaussian_data import CIFAR10_TRANSFORM
from src.gaussian_resnet import resnet20


def parse_args():
    p = argparse.ArgumentParser(
        description="Layerwise Fisher eigendecomposition / participation ratio "
        "for label-noise models (corrupted vs uncorrupted split)"
    )
    p.add_argument(
        "--n",
        type=int,
        required=True,
        help="Layer depth index selecting which Conv/Linear layer "
        "to analyze (and which fisher_matrix_{n}.pt to use)",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "output/noisy10pct"),
        help="Noisy-model directory containing model.pt and " "noise_metadata.npz",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data"),
        help="CIFAR-10 data directory",
    )
    p.add_argument(
        "--max-cifar",
        type=int,
        default=None,
        help="Cap on CIFAR-10 train samples (default: all 50000; "
        "for testing). Also caps the on-the-fly Fisher.",
    )
    p.add_argument(
        "--fisher-batch-size",
        type=int,
        default=256,
        help="Batch size for on-the-fly layerwise Fisher computation",
    )
    p.add_argument(
        "--recompute-fisher",
        action="store_true",
        help="Recompute the layerwise Fisher even if a cached " "fisher_matrix_{n}.pt exists",
    )
    p.add_argument(
        "--no-cache-fisher",
        action="store_true",
        help="Do not write the computed Fisher matrix to disk",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (default: cuda if available, else cpu)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Print a progress line every this many samples",
    )
    return p.parse_args()


def get_fisher_matrix(
    model, depth, model_dir, data_loader, device, K, recompute=False, cache=True, max_batches=None
):
    """Return the K x K layerwise Fisher for ``depth`` (CPU tensor).

    Loads ``fisher_matrix_{depth}.pt`` from ``model_dir`` if present (and
    ``recompute`` is False); otherwise computes the empirical layerwise Fisher
    via ``layerwise_fisher_wrt_weights`` over ``data_loader`` and (optionally)
    caches it to disk for reuse.
    """
    fisher_path = os.path.join(model_dir, f"fisher_matrix_{depth}.pt")

    if os.path.isfile(fisher_path) and not recompute:
        print(f"Loading cached Fisher matrix: {fisher_path}")
        fisher = torch.load(fisher_path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(fisher):
            raise TypeError(f"Expected a tensor in {fisher_path}, got {type(fisher)}")
        if fisher.dim() != 2 or fisher.shape[0] != fisher.shape[1]:
            raise ValueError(f"Expected a square matrix, got shape {tuple(fisher.shape)}")
        if fisher.shape[0] != K:
            raise ValueError(
                f"Cached Fisher size {fisher.shape[0]} != layer params {K}; "
                f"check that --n matches the layer used to build the matrix "
                f"(use --recompute-fisher to rebuild)."
            )
        return fisher

    print(f"Computing layerwise Fisher for depth {depth} " f"(no cache or --recompute-fisher)...")
    t_fisher = time.time()
    fisher_by_depth, _ = layerwise_fisher_wrt_weights(
        model, data_loader, device, max_batches=max_batches, target_depths={depth}
    )
    fisher = fisher_by_depth[depth]
    print(f"Fisher computation took {time.time() - t_fisher:.1f}s")

    if cache:
        torch.save(fisher, fisher_path)
        print(f"Cached Fisher matrix to {fisher_path}")
    return fisher


def eigendecompose_top_first(fisher, device):
    """Eigendecompose a symmetric Fisher matrix, returning TOP-FIRST order.

    Reuses the scalable ``_eigh_symmetric`` (GPU ``eigh`` for small matrices,
    SciPy MRRR for large ones), which returns ascending eigenvalues, then
    flips to descending so index/column 0 is the leading eigenvector k=1
    (matching the convention used in ``pr_gaussian.py``).
    """
    evals, evecs = _eigh_symmetric(fisher, device)  # ascending
    return evals.flip(0), evecs.flip(1)  # -> descending (top-first)


def single_sample_layer_fisher_gram(model, image, weight_param, device):
    """Gram factor G (K x C) of the single-sample Fisher restricted to one layer.

    The single-sample Fisher restricted to one layer's K weights is the
    K x K matrix  F_i = sum_c p(c|x) g_c g_c^T = G G^T, with columns of G
    equal to sqrt(p(c|x)) * grad_{w_layer} log p(c|x). F_i is PSD with rank
    <= C-1 = 9. We return the small K x C Gram factor rather than the full
    K x K matrix; quadratic forms <u|F_i|u> = ||G^T u||^2 follow directly.
    """
    model.eval()
    x = image.unsqueeze(0) if image.dim() == 3 else image
    x = x.to(device)

    logits = model(x)
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp().detach()
    n_classes = logits.shape[1]

    scaled_grads = []
    for c in range(n_classes):
        model.zero_grad(set_to_none=True)
        (g,) = torch.autograd.grad(
            log_probs[0, c], weight_param, retain_graph=(c < n_classes - 1), create_graph=False
        )
        scaled_grads.append(torch.sqrt(probs[0, c]) * g.detach().reshape(-1))

    return torch.stack(scaled_grads, dim=1)  # (K, C)


def participation_ratios(s1_all, s1_uncorrupted, s1_corrupted, s2, n):
    """Per-direction participation ratios from accumulated sums.

    s1_*  : sum_i c_i^k  over (all / uncorrupted / corrupted) samples  (K,)
    s2    : sum_i (c_i^k)^2  over all samples                          (K,)
    n     : total number of samples
    """
    denom = (n * s2).clamp(min=1e-30)
    return {
        "psi_all": (s1_all**2) / denom,
        "psi_uncorrupted": (s1_uncorrupted**2) / denom,
        "psi_corrupted": (s1_corrupted**2) / denom,
    }


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Model dir:  {args.model_dir}")
    print(f"Layer (n):  {args.n}")
    print(f"Data root:  {args.data_root}")
    print(f"Device:     {device}")

    model_file = os.path.join(args.model_dir, "model.pt")
    model = resnet20(num_classes=10)
    load_model_weights(model_file, model, device=device)
    model.to(device).eval()

    layers = _get_targeted_layers(model)
    if args.n < 0 or args.n >= len(layers):
        raise ValueError(f"Layer depth {args.n} out of range [0, {len(layers) - 1}]")
    target = layers[args.n]
    weight_param = target["module"].weight
    K = target["n_params"]
    print(f"Targeted layer: depth={target['depth']}, name='{target['name']}', " f"K={K} params")

    meta_path = os.path.join(args.model_dir, "noise_metadata.npz")
    noise_meta = np.load(meta_path, allow_pickle=True)
    is_noised = np.asarray(noise_meta["is_noised"]).astype(bool)  # (50000,)
    noise_fraction = float(noise_meta["noise_fraction"])
    print(
        f"Noise fraction: {noise_fraction:.0%}  "
        f"(corrupted={int(is_noised.sum())}, "
        f"uncorrupted={int((~is_noised).sum())})"
    )

    cifar_train = datasets.CIFAR10(
        root=args.data_root, train=True, download=False, transform=CIFAR10_TRANSFORM
    )
    n_cifar = len(cifar_train) if args.max_cifar is None else min(args.max_cifar, len(cifar_train))
    if n_cifar > len(is_noised):
        raise ValueError(f"n_cifar={n_cifar} exceeds noise mask length {len(is_noised)}")
    n_total = n_cifar
    n_corrupted = int(is_noised[:n_cifar].sum())
    n_uncorrupted = n_total - n_corrupted
    print(
        f"Samples: {n_total} CIFAR train "
        f"({n_uncorrupted} uncorrupted + {n_corrupted} corrupted)"
    )

    fisher_loader = DataLoader(
        cifar_train, batch_size=args.fisher_batch_size, shuffle=False, num_workers=0
    )
    max_fisher_batches = (
        None if args.max_cifar is None else int(np.ceil(n_cifar / args.fisher_batch_size))
    )
    fisher = get_fisher_matrix(
        model,
        args.n,
        args.model_dir,
        fisher_loader,
        device,
        K,
        recompute=args.recompute_fisher,
        cache=not args.no_cache_fisher,
        max_batches=max_fisher_batches,
    )
    print(f"Fisher matrix: shape={tuple(fisher.shape)}, dtype={fisher.dtype}")

    t0 = time.time()
    evals, evecs = eigendecompose_top_first(fisher, device)
    del fisher
    print(f"Eigendecomposition ({K}x{K}) took {time.time() - t0:.1f}s")
    print(f"Eigenvalues (top-first): max={evals[0].item():.3e}, " f"min={evals[-1].item():.3e}")
    evecs = evecs.to(device=device, dtype=weight_param.dtype)  # (K, K)

    s1_uncorrupted = torch.zeros(K, device=device)  # sum_i c_i^k over uncorrupted
    s1_corrupted = torch.zeros(K, device=device)  # sum_i c_i^k over corrupted
    s2 = torch.zeros(K, device=device)  # sum_i (c_i^k)^2 over all

    print("Computing single-sample Fisher overlaps...")
    t_loop = time.time()
    for i in range(n_total):
        image, _ = cifar_train[i]
        is_corrupted = bool(is_noised[i])

        G = single_sample_layer_fisher_gram(model, image, weight_param, device)
        M = evecs.T @ G  # (K, C); row k = u_k^T G
        c = (M * M).sum(dim=1)  # c_i^k = ||G^T u_k||^2  (K,)

        s2 += c * c
        if is_corrupted:
            s1_corrupted += c
        else:
            s1_uncorrupted += c

        if (i + 1) % args.progress_every == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_loop
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate
            print(
                f"  [{i + 1}/{n_total}] {elapsed:.0f}s elapsed, "
                f"{rate:.1f} samp/s, ~{eta:.0f}s remaining"
            )

    s1_all = s1_uncorrupted + s1_corrupted

    lambda_check = (s1_all / n_total).cpu()
    evals_cpu = evals.cpu()
    max_abs_err = (lambda_check - evals_cpu).abs().max().item()
    denom_lambda = evals_cpu.abs().max().clamp(min=1e-30).item()
    print("\nSanity check (1/n) sum_i c_i^k  vs  lambda_k:")
    print(
        f"  max |diff| = {max_abs_err:.3e}  "
        f"(relative to max|lambda| = {max_abs_err / denom_lambda:.3e})"
    )

    psi = participation_ratios(s1_all, s1_uncorrupted, s1_corrupted, s2, n_total)
    psi = {k: v.cpu() for k, v in psi.items()}

    print(f"\nParticipation ratio summary (K={K} directions, n={n_total}):")
    for name in ("psi_all", "psi_uncorrupted", "psi_corrupted"):
        v = psi[name]
        print(
            f"  {name:16s}: mean={v.mean().item():.4f}, "
            f"k=1(top)={v[0].item():.4f}, k=K(bottom)={v[-1].item():.4f}"
        )

    out = {
        "n": args.n,
        "layer_name": target["name"],
        "K": K,
        "noise_fraction": noise_fraction,
        "n_uncorrupted": n_uncorrupted,
        "n_corrupted": n_corrupted,
        "n_total": n_total,
        "eigenvalues": evals_cpu,  # top-first (k=1 .. k=K)
        "lambda_check": lambda_check,  # (1/n) sum_i c_i^k
        "lambda_max_abs_err": max_abs_err,
        "s1_all": s1_all.cpu(),  # sum_i c_i^k
        "s1_uncorrupted": s1_uncorrupted.cpu(),
        "s1_corrupted": s1_corrupted.cpu(),
        "s2": s2.cpu(),  # sum_i (c_i^k)^2
        "psi_all": psi["psi_all"],
        "psi_uncorrupted": psi["psi_uncorrupted"],
        "psi_corrupted": psi["psi_corrupted"],
    }
    save_prefix = os.path.join(args.model_dir, f"PR_noisy_layerwise_{args.n}")
    save_experiment_data(out, save_prefix + ".pt")
    print(f"\nSaved results to {save_prefix}.pt")

    return out


if __name__ == "__main__":
    main()
