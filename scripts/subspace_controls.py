"""Fisher vs weight-SVD vs random subspaces (paper Fig. 11)."""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.utils.data as tud

import _bootstrap  # noqa: F401
from src.fisher_utils import (
    cache_eval_splits,
    cache_split,
    check_draw_independence,
    complete_basis,
    descending_eigh,
    eval_cached,
    get_jac_layer,
    get_layer_params,
    k_schedule,
    load_resnet20,
    refresh_bn_cached,
)
from src.label_noise_data import NUM_CLASSES, SPLITS, load_experiment


def fisher_basis(model, jac_loader, device, C, N, layer_name):
    """Eigenvectors of G = J J^T, ordered by decreasing eigenvalue."""
    t0 = time.time()
    jac = get_jac_layer(model, jac_loader, device, C, N, layer_name)
    print(
        f"  Jacobian {jac.shape} in {time.time() - t0:.1f}s " f"({jac.nbytes / 1e9:.2f} GB)",
        flush=True,
    )

    t0 = time.time()
    jac_gpu = torch.from_numpy(jac).to(device)
    del jac
    G = jac_gpu @ jac_gpu.T
    del jac_gpu
    evals, U, rank = descending_eigh(G)
    del G
    print(f"  Eigendecomp in {time.time() - t0:.1f}s, rank={rank:,}/{U.shape[1]:,}", flush=True)

    U, n_repaired = complete_basis(U, device)
    return U, evals, rank, n_repaired


def random_basis(P, seed, device):
    """Haar-distributed orthonormal basis of R^P via QR of a Gaussian matrix.

    float32 rather than float64: a random basis carries no small eigenvalue
    structure to resolve, and the square QR would otherwise need three P x P
    float64 buffers (~33 GB at P = 36,864).
    """
    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(seed)
    M = torch.randn(P, P, generator=gen, device=device, dtype=torch.float32)
    Q, R = torch.linalg.qr(M)
    del M
    Q *= torch.sign(torch.diagonal(R)).unsqueeze(0)  # exactly Haar, not QR-biased
    del R
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"  Random basis ({P:,} x {P:,}) in {time.time() - t0:.1f}s", flush=True)
    return Q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None, help="Default: <model-dir>/controls")
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument("--method", choices=["fisher", "svd", "random"], required=True)
    parser.add_argument("--fisher-set", choices=["clean", "full"], default="clean")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--n-k-points", type=int, default=400)
    parser.add_argument("--jac-bs", type=int, default=8)
    parser.add_argument("--eval-bs", type=int, default=500)
    parser.add_argument("--n-eval-batches", type=int, default=10)
    parser.add_argument("--n-bn-batches", type=int, default=10)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.model_dir, "controls")
    os.makedirs(out_dir, exist_ok=True)
    device, C = args.device, NUM_CLASSES

    tag = args.prefix or (f"fisher_{args.fisher_set}" if args.method == "fisher" else args.method)
    stem = os.path.join(out_dir, f"{tag}_{args.layer.replace('.', '_')}_seed{args.seed}")
    print(
        f"=== layer={args.layer} method={args.method} "
        f"fisher-set={args.fisher_set} seed={args.seed} ===",
        flush=True,
    )

    model = load_resnet20(f"{args.model_dir}/model.pt", device)
    exp = load_experiment(args.model_dir, args.data_root)

    layer_params, w0_np = get_layer_params(model, args.layer)
    P = w0_np.size
    print(
        f"Clean: {len(exp['clean_idx'])}, Noised: {len(exp['noised_idx'])} "
        f"({100 * exp['noise_fraction']:.1f}% noise), P={P:,}",
        flush=True,
    )

    # The eval and BN subsets are drawn from the seed, so the band includes
    # evaluation noise rather than only subspace-construction noise.
    rng = np.random.RandomState(args.seed)

    def shuffled(idx):
        idx = np.asarray(idx).copy()
        rng.shuffle(idx)
        return idx

    t0 = time.time()
    cached = cache_eval_splits(
        exp["splits"], args.n_eval_batches, args.eval_bs, device, order=shuffled
    )
    bn_batches = cache_split(
        exp["train_noised"],
        shuffled(np.arange(len(exp["train_noised"]))),
        args.n_bn_batches,
        1000,
        device,
    )
    print(f"Cached evaluation splits in {time.time() - t0:.1f}s", flush=True)

    baseline = {name: eval_cached(model, batches) for name, batches in cached.items()}
    print("Baseline acc: " + "  ".join(f"{n}={baseline[n][1]:.4f}" for n in SPLITS), flush=True)

    dtype = next(model.parameters()).dtype
    basis_dtype = torch.float32 if args.method == "random" else torch.float64
    basis_tol = 1e-3 if args.method == "random" else 1e-6
    w0 = torch.from_numpy(w0_np).to(device=device, dtype=basis_dtype)

    U = evals = Uw = S = weight_shape = None
    fisher_rank, n_repaired = -1, 0

    if args.method == "fisher":
        N_needed = P // C + 1
        jac_pool = exp["clean_idx"] if args.fisher_set == "clean" else np.arange(len(exp["train"]))
        jac_idx = rng.choice(jac_pool, size=min(N_needed, len(jac_pool)), replace=False)
        check_draw_independence(jac_idx, jac_pool, exp["is_noised"])
        jac_loader = tud.DataLoader(
            tud.Subset(exp["train_noised"], jac_idx.tolist()),
            batch_size=args.jac_bs,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        print(f"Fisher from {args.fisher_set} samples ({len(jac_idx):,})", flush=True)
        U, evals, fisher_rank, n_repaired = fisher_basis(
            model, jac_loader, device, C, N_needed, args.layer
        )
    elif args.method == "random":
        U = random_basis(P, args.seed, device)
    else:
        mod = dict(model.named_modules())[args.layer]
        # The flat parameter vector is reshaped as a matrix to take its SVD,
        # which only makes sense if the sublayer is all weight. ResNet-20's
        # convolutions have bias=False; its linear head does not.
        if mod.weight.numel() != P:
            raise SystemExit(
                f"--method svd needs a bias-free sublayer; {args.layer} has "
                f"{P - mod.weight.numel()} extra parameters beyond its weight"
            )
        weight_shape = (mod.weight.shape[0], P // mod.weight.shape[0])
        t0 = time.time()
        Uw, S, _ = torch.linalg.svd(w0.view(*weight_shape), full_matrices=False)
        print(
            f"  Weight SVD {weight_shape} in {time.time() - t0:.1f}s, "
            f"{S.numel()} singular values",
            flush=True,
        )

    if args.method == "svd":
        # A rank-r truncation projects onto the span of r left singular
        # vectors, a linear subspace of dimension r * n_cols.
        k_ranks = np.arange(1, min(weight_shape) + 1)
        k_values = k_ranks * weight_shape[1]
        W0 = w0.view(*weight_shape)

        def project(r):
            Ur = Uw[:, :r]
            return (Ur @ (Ur.T @ W0)).reshape(-1)

        def gram_err():
            I = torch.eye(Uw.shape[1], device=device, dtype=Uw.dtype)
            return float((Uw.T @ Uw - I).abs().max())

        n_full = min(weight_shape)
    else:
        k_values = np.array(k_schedule(P, args.n_k_points, C))
        k_ranks = k_values.copy()

        def project(k):
            Uk = U[:, :k]
            return Uk @ (Uk.T @ w0)

        # Orthonormality on a random column sample; the full P x P Gram matrix
        # is too large to form for the deep layers.
        n_chk = min(1024, P)
        chk = torch.randperm(P, device=device)[:n_chk]

        def gram_err():
            Uc = U[:, chk]
            I = torch.eye(n_chk, device=device, dtype=Uc.dtype)
            return float((Uc.T @ Uc - I).abs().max())

        n_full = P

    recon_err = float((project(n_full) - w0).norm() / w0.norm())
    gerr = gram_err()
    print(
        f"  {args.method} completeness: ||B B^T w0 - w0||/||w0|| = {recon_err:.3e}, "
        f"max|B^T B - I| = {gerr:.3e} (tol {basis_tol:.0e})",
        flush=True,
    )
    if args.method != "svd" and max(recon_err, gerr) > basis_tol:
        raise SystemExit(f"Basis for {args.layer} is not complete and orthonormal")

    n_pts = len(k_values)
    print(
        f"{n_pts} eval points, k_dim in [{k_values[0]:,}..{k_values[-1]:,}] " f"of P={P:,}",
        flush=True,
    )

    def write_weights(vec):
        idx = 0
        for _, p in layer_params:
            n = p.numel()
            p.data.copy_(vec[idx : idx + n].view(p.shape).to(dtype))
            idx += n

    losses = {n: [] for n in SPLITS}
    accs = {n: [] for n in SPLITS}
    norm_retained = []
    w0_norm = float(w0.norm())

    t0 = time.time()
    for i, (k, r) in enumerate(zip(k_values, k_ranks)):
        proj = project(int(r) if args.method == "svd" else int(k))
        norm_retained.append(float(proj.norm()) / w0_norm)
        write_weights(proj)
        refresh_bn_cached(model, bn_batches)
        for name in SPLITS:
            loss, acc = eval_cached(model, cached[name])
            losses[name].append(loss)
            accs[name].append(acc)
        if i % 20 == 0 or i == n_pts - 1:
            el = time.time() - t0
            print(
                f"  [{i+1}/{n_pts}] k_dim={k:,} "
                f"train={accs['clean_train'][-1]:.3f} "
                f"flipped={accs['noised_noisedlbl'][-1]:.3f} "
                f"true={accs['noised_truelbl'][-1]:.3f} "
                f"test={accs['test'][-1]:.3f} | {el:.0f}s, "
                f"ETA {el / (i + 1) * (n_pts - i - 1):.0f}s",
                flush=True,
            )
    print(f"Sweep in {time.time() - t0:.1f}s", flush=True)

    write_weights(w0)

    save = {
        "k_values": np.asarray(k_values),
        "k_rank": np.asarray(k_ranks),
        "weight_norm_retained": np.array(norm_retained),
        "P": P,
        "method": args.method,
        "layer": args.layer,
        "seed": args.seed,
        "fisher_set": args.fisher_set if args.method == "fisher" else "none",
        "noise_fraction": exp["noise_fraction"],
        "fisher_rank": fisher_rank,
        "n_repaired_columns": n_repaired,
        "basis_recon_err": recon_err,
        "basis_gram_err": gerr,
    }
    for name in SPLITS:
        save[name] = np.array(losses[name])
        save[name + "_acc"] = np.array(accs[name])
        save["baseline_" + name] = baseline[name][0]
        save["baseline_" + name + "_acc"] = baseline[name][1]
    if args.method == "fisher":
        save["eigenvalues"] = evals.cpu().numpy()
    if args.method == "svd":
        save["singular_values"] = S.cpu().numpy()
        save["weight_shape"] = np.array(weight_shape)

    np.savez(stem + ".npz", **save)
    with open(stem + "_config.json", "w") as f:
        json.dump({**vars(args), "P": int(P), "n_eval_points": int(n_pts)}, f, indent=2)

    if str(device).startswith("cuda"):
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Saved: {stem}.npz\nDone.", flush=True)


if __name__ == "__main__":
    main()
