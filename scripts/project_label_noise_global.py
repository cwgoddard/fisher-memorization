"""Block-diagonal Fisher projection across all parameters (paper Fig. 3)."""

import argparse
import os
import time

import numpy as np
import torch
import torch.utils.data as tud
from tqdm import tqdm

import _bootstrap  # noqa: F401
from src.fisher_utils import (
    cache_eval_splits,
    cache_split,
    complete_basis,
    descending_eigh,
    draw_fisher_samples,
    enumerate_sublayers,
    eval_cached,
    get_jac_layer,
    get_layer_params,
    k_schedule,
    load_resnet20,
    refresh_bn_cached,
    set_layer_weights,
)
from src.label_noise_data import NUM_CLASSES, SPLITS, load_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--fisher-set", choices=["clean", "full"], default="clean")
    parser.add_argument("--prefix", type=str, default="global")
    parser.add_argument(
        "--reverse", action="store_true", help="Eigendecompose largest blocks first"
    )
    parser.add_argument("--n-k-points", type=int, default=500)
    parser.add_argument("--jac-bs", type=int, default=8)
    parser.add_argument("--eval-bs", type=int, default=500)
    parser.add_argument("--n-eval-batches", type=int, default=10)
    parser.add_argument("--n-bn-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = args.out_dir or args.model_dir
    os.makedirs(out_dir, exist_ok=True)
    device, C = args.device, NUM_CLASSES

    model = load_resnet20(f"{args.model_dir}/model.pt", device)
    exp = load_experiment(args.model_dir, args.data_root)
    print(
        f"Clean: {len(exp['clean_idx'])}, Noised: {len(exp['noised_idx'])} "
        f"({100 * exp['noise_fraction']:.1f}% noise)"
    )

    t0 = time.time()
    cached = cache_eval_splits(exp["splits"], args.n_eval_batches, args.eval_bs, device)
    bn_batches = cache_split(
        exp["train_noised"], np.arange(len(exp["train_noised"])), args.n_bn_batches, 1000, device
    )
    print(f"Cached evaluation splits in {time.time() - t0:.1f}s")

    sublayers = enumerate_sublayers(model)
    total_P = sum(P for _, P in sublayers)
    max_N = max(P // C + 1 for _, P in sublayers)
    print(f"\n{len(sublayers)} blocks, total P = {total_P:,}, max N = {max_N:,}")

    jac_pool = exp["clean_idx"] if args.fisher_set == "clean" else np.arange(len(exp["train"]))
    jac_idx = draw_fisher_samples(jac_pool, max_N, args.seed, exp["is_noised"])
    jac_loader = tud.DataLoader(
        tud.Subset(exp["train_noised"], jac_idx.tolist()),
        batch_size=args.jac_bs,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    n_corrupt = int(exp["is_noised"][jac_idx].sum())
    print(
        f"Fisher from {args.fisher_set} samples "
        f"({len(jac_idx):,} total, {n_corrupt:,} corrupted)"
    )

    original_sd = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\n{'=' * 60}\nPer-block eigendecompositions\n{'=' * 60}")
    eigen_order = (
        sorted(sublayers, key=lambda x: x[1], reverse=True) if args.reverse else list(sublayers)
    )

    block_eigenvalues, block_U, block_coeffs = {}, {}, {}
    total_mem = 0
    for layer_name, P_layer in eigen_order:
        N_needed = P_layer // C + 1
        print(f"\n{layer_name} | P={P_layer:,} | N={N_needed:,}")
        model.load_state_dict(original_sd)

        t0 = time.time()
        jac = get_jac_layer(model, jac_loader, device, C, N_needed, layer_name)
        print(f"  Jacobian {jac.shape} in {time.time() - t0:.1f}s")

        t0 = time.time()
        jac_gpu = torch.from_numpy(jac).to(device)
        del jac
        G = jac_gpu @ jac_gpu.T
        del jac_gpu
        eigenvalues, eigvecs, rank = descending_eigh(G)
        del G
        print(f"  Eigendecomp in {time.time() - t0:.1f}s, rank={rank:,}/{P_layer:,}")

        eigvecs, n_repaired = complete_basis(eigvecs, device)
        U = eigvecs.cpu().numpy()
        eig_np = eigenvalues.clamp(min=0).cpu().numpy()
        del eigvecs, eigenvalues
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        _, w0 = get_layer_params(model, layer_name)
        coeffs = U.T @ w0
        recon_err = float(np.linalg.norm(U @ coeffs - w0) / np.linalg.norm(w0))
        print(
            f"  Reconstruction error at k=P: {recon_err:.2e}"
            + (f"  (repaired {n_repaired:,} columns)" if n_repaired else "")
        )
        if recon_err > 1e-8:
            print(f"  WARNING: basis does not span R^P for {layer_name}")

        block_eigenvalues[layer_name] = eig_np
        block_U[layer_name] = U
        block_coeffs[layer_name] = coeffs
        total_mem += U.nbytes + coeffs.nbytes
        print(f"  Stored U {U.shape} float64, host total {total_mem / 1e9:.1f} GB")

    model.load_state_dict(original_sd)

    entries = [(ev, name) for name, eigs in block_eigenvalues.items() for ev in eigs]
    entries.sort(key=lambda x: x[0], reverse=True)
    total_rank = len(entries)
    print(f"\nPooled {total_rank:,} eigenvalues across {len(sublayers)} blocks")

    # cumulative_alloc[name][k] = how many of the global top-k modes are this
    # block's, so the sweep only has to look up one integer per block per k.
    block_names = [name for name, _ in sublayers]
    cumulative_alloc = {name: np.zeros(total_rank + 1, dtype=np.int32) for name in block_names}
    for pos, (_, layer_name) in enumerate(entries):
        for name in block_names:
            cumulative_alloc[name][pos + 1] = cumulative_alloc[name][pos]
        cumulative_alloc[layer_name][pos + 1] += 1

    ks = k_schedule(total_rank, args.n_k_points, C)
    print(f"{len(ks)} eval points, k in [{ks[0]}..{ks[-1]}]")

    dtype = next(model.parameters()).dtype
    block_layer_params, block_w0 = {}, {}
    for layer_name, _ in sublayers:
        lp, w0 = get_layer_params(model, layer_name)
        block_layer_params[layer_name] = lp
        block_w0[layer_name] = w0

    losses = {n: [] for n in SPLITS}
    accs = {n: [] for n in SPLITS}

    t0 = time.time()
    for k in tqdm(ks, desc="Global sweep"):
        for layer_name, _ in sublayers:
            k_i = int(cumulative_alloc[layer_name][k])
            lp = block_layer_params[layer_name]
            if k_i == 0:
                set_layer_weights(lp, np.zeros_like(block_w0[layer_name]), dtype, device)
            elif k_i >= block_U[layer_name].shape[1]:
                set_layer_weights(lp, block_w0[layer_name], dtype, device)
            else:
                U_i = block_U[layer_name]
                set_layer_weights(lp, U_i[:, :k_i] @ block_coeffs[layer_name][:k_i], dtype, device)
        refresh_bn_cached(model, bn_batches)
        for name in SPLITS:
            loss, acc = eval_cached(model, cached[name])
            losses[name].append(loss)
            accs[name].append(acc)
    print(f"Sweep in {time.time() - t0:.1f}s")

    model.load_state_dict(original_sd)

    save = {
        "k_values": np.array(ks),
        "total_P": total_P,
        "total_rank": total_rank,
        "fisher_set": args.fisher_set,
        "seed": args.seed,
        "noise_fraction": exp["noise_fraction"],
        "n_jac_samples": len(jac_idx),
        "n_jac_corrupted": n_corrupt,
    }
    for name in SPLITS:
        save[name] = np.array(losses[name])
        save[name + "_acc"] = np.array(accs[name])
    npz_path = os.path.join(out_dir, f"{args.prefix}_proj.npz")
    np.savez(npz_path, **save)
    print(f"\nSaved: {npz_path}\nDone.")


if __name__ == "__main__":
    main()
