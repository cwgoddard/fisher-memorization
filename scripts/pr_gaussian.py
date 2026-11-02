"""Layerwise Fisher participation-ratio analysis for CIFAR-10 + Gaussian models."""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torchvision import datasets

from _bootstrap import PROJECT_ROOT
from src.experiment_utils import load_model_weights, save_experiment_data
from src.fisher_projection import _eigh_symmetric, _get_targeted_layers
from src.gaussian_data import CIFAR10_TRANSFORM
from src.gaussian_resnet import resnet20


def parse_args():
    p = argparse.ArgumentParser(
        description="Layerwise Fisher eigendecomposition / participation ratio"
    )
    p.add_argument(
        "--n",
        type=int,
        required=True,
        help="Layer depth index selecting which fisher_matrix_{n}.pt to load",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "experiments/resnet20_cifar10gaussian_g500_randlabels_run004"
        ),
        help="Experiment directory containing fisher_matrix_{n}.pt files",
    )
    p.add_argument(
        "--max-cifar",
        type=int,
        default=None,
        help="Cap on CIFAR-10 train samples (default: all 50000; for testing)",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="CIFAR-10 data directory (default: read from model config)",
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


def load_fisher_matrix(model_path, n):
    """Load the layerwise Fisher matrix for layer depth ``n`` (K x K, CPU)."""
    fisher_path = os.path.join(model_path, f"fisher_matrix_{n}.pt")
    if not os.path.isfile(fisher_path):
        raise FileNotFoundError(f"No Fisher matrix found at {fisher_path}")

    fisher = torch.load(fisher_path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(fisher):
        raise TypeError(f"Expected a tensor in {fisher_path}, got {type(fisher)}")
    if fisher.dim() != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {tuple(fisher.shape)}")
    return fisher


def eigh_symmetric(fisher, device):
    """Eigendecompose a symmetric Fisher matrix in TOP-FIRST order.

    The shared solver returns the ``torch.linalg.eigh`` convention
    (ascending eigenvalues) for every matrix size. Flip exactly once here so
    saved Gaussian PR outputs use the same convention as ``pr_label_noise.py``:
    index/column 0 is the leading Fisher eigenvector.
    """
    evals, evecs = _eigh_symmetric(fisher, device)  # ascending
    return evals.flip(0), evecs.flip(1)  # -> descending


def single_sample_layer_fisher_gram(model, image, weight_param, device):
    """Gram factor G (K x C) of the single-sample Fisher restricted to one layer.

    The single-sample Fisher restricted to one layer's K weights is the
    K x K matrix  F_i = sum_c p(c|x) g_c g_c^T = G G^T, with columns of G
    equal to sqrt(p(c|x)) * grad_{w_layer} log p(c|x). F_i is PSD with rank
    <= C-1 = 9. We return the small K x C Gram factor rather than the full
    K x K matrix; quadratic forms <u|F_i|u> = ||G^T u||^2 follow directly.
    This mirrors the rank-9 trick in ``full_model_single_sample_fisher_top_k``
    but restricted to a single layer.
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


def participation_ratios(s1_all, s1_cifar, s1_gauss, s2, n):
    """Per-direction participation ratios from accumulated sums.

    s1_*  : sum_i c_i^k  over (all / cifar-only / gauss-only) samples  (K,)
    s2    : sum_i (c_i^k)^2  over all samples                          (K,)
    n     : total number of samples
    """
    denom = (n * s2).clamp(min=1e-30)
    return {
        "psi_all": (s1_all**2) / denom,
        "psi_cifar": (s1_cifar**2) / denom,
        "psi_gauss": (s1_gauss**2) / denom,
    }


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    config_file = os.path.join(args.model_path, "config.json")
    with open(config_file) as f:
        config = json.load(f)
    data_root = args.data_root or config.get("data_root", os.path.join(PROJECT_ROOT, "data"))

    print(f"Model path: {args.model_path}")
    print(f"Layer (n):  {args.n}")
    print(f"Data root:  {data_root}")
    print(f"Device:     {device}")

    model_file = os.path.join(args.model_path, "final_model.pt")
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

    fisher = load_fisher_matrix(args.model_path, args.n)
    if fisher.shape[0] != K:
        raise ValueError(
            f"Fisher matrix size {fisher.shape[0]} != layer params {K}; "
            f"check that --n matches the layer used to build the matrix."
        )
    print(f"Loaded Fisher matrix: shape={tuple(fisher.shape)}, dtype={fisher.dtype}")

    t0 = time.time()

    evals, evecs = eigh_symmetric(fisher, device)

    del fisher
    print(f"Eigendecomposition ({K}x{K}) took {time.time() - t0:.1f}s")
    print(f"Eigenvalues (top-first): max={evals[0].item():.3e}, " f"min={evals[-1].item():.3e}")
    evecs = evecs.to(device=device, dtype=weight_param.dtype)  # (K, K)

    cifar_train = datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=CIFAR10_TRANSFORM
    )
    n_cifar = len(cifar_train) if args.max_cifar is None else min(args.max_cifar, len(cifar_train))

    gauss_meta = torch.load(
        os.path.join(args.model_path, "gaussian_metadata.pt"), weights_only=False
    )
    gauss_images = gauss_meta["images"]
    n_gauss = len(gauss_images)

    n_total = n_cifar + n_gauss
    print(f"Samples: {n_cifar} CIFAR + {n_gauss} Gaussian = {n_total} total")

    s1_cifar = torch.zeros(K, device=device)  # sum_i c_i^k over CIFAR
    s1_gauss = torch.zeros(K, device=device)  # sum_i c_i^k over Gaussian
    s2 = torch.zeros(K, device=device)  # sum_i (c_i^k)^2 over all

    def iter_samples():
        for i in range(n_cifar):
            img, _ = cifar_train[i]
            yield img, False  # is_gauss = False
        for j in range(n_gauss):
            yield gauss_images[j], True  # is_gauss = True

    print("Computing single-sample Fisher overlaps...")
    t_loop = time.time()
    for i, (image, is_gauss) in enumerate(iter_samples()):
        G = single_sample_layer_fisher_gram(model, image, weight_param, device)
        M = evecs.T @ G  # (K, C); row k = u_k^T G
        c = (M * M).sum(dim=1)  # c_i^k = ||G^T u_k||^2  (K,)

        s2 += c * c
        if is_gauss:
            s1_gauss += c
        else:
            s1_cifar += c

        if (i + 1) % args.progress_every == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_loop
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate
            print(
                f"  [{i + 1}/{n_total}] {elapsed:.0f}s elapsed, "
                f"{rate:.1f} samp/s, ~{eta:.0f}s remaining"
            )

    s1_all = s1_cifar + s1_gauss

    lambda_check = (s1_all / n_total).cpu()
    evals_cpu = evals.cpu()
    max_abs_err = (lambda_check - evals_cpu).abs().max().item()
    denom_lambda = evals_cpu.abs().max().clamp(min=1e-30).item()
    print("\nSanity check (1/n) sum_i c_i^k  vs  lambda_k:")
    print(
        f"  max |diff| = {max_abs_err:.3e}  "
        f"(relative to max|lambda| = {max_abs_err / denom_lambda:.3e})"
    )

    psi = participation_ratios(s1_all, s1_cifar, s1_gauss, s2, n_total)
    psi = {k: v.cpu() for k, v in psi.items()}

    print(f"\nParticipation ratio summary (K={K} directions, n={n_total}):")
    for name in ("psi_all", "psi_cifar", "psi_gauss"):
        v = psi[name]
        print(
            f"  {name:10s}: mean={v.mean().item():.4f}, "
            f"k=1(top)={v[0].item():.4f}, k=K(bottom)={v[-1].item():.4f}"
        )

    out = {
        "n": args.n,
        "layer_name": target["name"],
        "K": K,
        "n_cifar": n_cifar,
        "n_gauss": n_gauss,
        "n_total": n_total,
        "eigenvalues": evals_cpu,  # top-first (k=1 .. k=K)
        "lambda_check": lambda_check,  # (1/n) sum_i c_i^k
        "lambda_max_abs_err": max_abs_err,
        "s1_all": s1_all.cpu(),  # sum_i c_i^k
        "s1_cifar": s1_cifar.cpu(),
        "s1_gauss": s1_gauss.cpu(),
        "s2": s2.cpu(),  # sum_i (c_i^k)^2
        "psi_all": psi["psi_all"],
        "psi_cifar": psi["psi_cifar"],
        "psi_gauss": psi["psi_gauss"],
    }
    save_prefix = os.path.join(args.model_path, f"PR_gauss_layerwise_{args.n}")
    save_experiment_data(out, save_prefix + ".pt")
    print(f"\nSaved results to {save_prefix}.pt")

    return out


if __name__ == "__main__":
    main()
