"""Reproduce the cuSOLVER eigh size limit that complete_basis repairs."""

import time

import torch


def synth(n, rank_frac=0.8, seed=0, device="cuda"):
    """A = (n, r) random, so G = A A^T is PSD with exact rank r < n."""
    g = torch.Generator(device=device).manual_seed(seed)
    r = int(rank_frac * n)
    A = torch.randn(n, r, generator=g, device=device, dtype=torch.float64)
    A /= n**0.5
    return A, r


def report_eigh(n, rank_frac=0.8):
    A, r = synth(n, rank_frac)
    torch.cuda.synchronize()
    t0 = time.time()
    G = A @ A.T
    evals, U = torch.linalg.eigh(G)
    torch.cuda.synchronize()
    t_eigh = time.time() - t0

    # flip to descending, as descending_eigh does
    evals = evals.flip(0).contiguous()
    U = U.flip(1).contiguous()

    colnorm = U.pow(2).sum(dim=0)  # should be 1.0 for every column
    n_zero = int((colnorm == 0).sum().item())
    n_bad = int((colnorm < 0.5).sum().item())
    lam_max = float(evals[0])

    # independent check of the same identity the pipeline uses
    sum1 = (U.T @ A).pow(2).sum(dim=1)
    denom = evals.clamp(min=lam_max * 1e-12)
    rel = (sum1 - evals).abs() / denom
    well = evals > lam_max * 1e-8

    print(f"\n  eigh n={n:,} (true rank {r:,}) in {t_eigh:.1f}s")
    print(f"    zero-norm U columns : {n_zero:,}     (n - n_zero = {n - n_zero:,})")
    print(f"    ||u_k|| < 0.5       : {n_bad:,}")
    print(f"    max |1 - ||u_k|| |  : {float((colnorm - 1).abs().max()):.3e}")
    print(f"    max rel err, all    : {float(rel.max()):.3e}")
    print(
        f"    max rel err, eig>1e-8*lmax ({int(well.sum()):,} modes): "
        f"{float(rel[well].max()):.3e}"
    )
    del A, G, U, evals, sum1
    torch.cuda.empty_cache()
    return n_zero


def report_svd(n, rank_frac=0.8):
    """Same matrix, but get the eigenbasis from an SVD of A instead of eigh(G)."""
    A, r = synth(n, rank_frac)
    torch.cuda.synchronize()
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    torch.cuda.synchronize()
    t_svd = time.time() - t0

    colnorm = U.pow(2).sum(dim=0)
    n_zero = int((colnorm == 0).sum().item())
    evals = S.pow(2)
    lam_max = float(evals[0])
    # For the SVD path U^T A = S V^T is already available, no second matmul.
    sum1 = (S[:, None] * Vh).pow(2).sum(dim=1)
    denom = evals.clamp(min=lam_max * 1e-12)
    rel = (sum1 - evals).abs() / denom
    well = evals > lam_max * 1e-8

    print(f"\n  svd  n={n:,} (true rank {r:,}) in {t_svd:.1f}s   U shape {tuple(U.shape)}")
    print(f"    zero-norm U columns : {n_zero:,}")
    print(f"    max |1 - ||u_k|| |  : {float((colnorm - 1).abs().max()):.3e}")
    print(f"    max rel err, all    : {float(rel.max()):.3e}")
    print(
        f"    max rel err, eig>1e-8*lmax ({int(well.sum()):,} modes): "
        f"{float(rel[well].max()):.3e}"
    )
    del A, U, S, Vh, evals, sum1
    torch.cuda.empty_cache()
    return n_zero


def main():
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 70)
    print("eigh(G): sweep across the 2^15 = 32,768 boundary")
    print("=" * 70)
    for n in [8192, 32768, 32769, 36864]:
        try:
            report_eigh(n)
        except RuntimeError as e:
            print(f"\n  eigh n={n:,} FAILED: {e}")
        print(f"    peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()

    print("\n" + "=" * 70)
    print("svd(A): same matrices, alternative route to the eigenbasis")
    print("=" * 70)
    for n in [32768, 36864]:
        try:
            report_svd(n)
        except RuntimeError as e:
            print(f"\n  svd n={n:,} FAILED: {e}")
        print(f"    peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()

    print("\nDone.")


if __name__ == "__main__":
    main()
