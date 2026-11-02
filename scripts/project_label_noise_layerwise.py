"""Per-sublayer Fisher projection sweeps (paper Fig. 2 and the all-layer appendix grids)."""

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
    select_sublayers,
    set_layer_weights,
)
from src.label_noise_data import NUM_CLASSES, SPLITS, load_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory with model.pt and noise_metadata.npz",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--fisher-set",
        choices=["clean", "full"],
        default="clean",
        help="Which training samples estimate the Fisher",
    )
    parser.add_argument("--prefix", type=str, default="proj")
    parser.add_argument(
        "--layers", type=str, default=None, help="Comma-separated sublayer names (default: all)"
    )
    parser.add_argument("--reverse", action="store_true", help="Process largest sublayers first")
    parser.add_argument("--n-k-points", type=int, default=100)
    parser.add_argument("--jac-bs", type=int, default=8)
    parser.add_argument("--eval-bs", type=int, default=500)
    parser.add_argument("--n-eval-batches", type=int, default=10)
    parser.add_argument("--n-bn-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="Seeds the Fisher sample draw")
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

    sublayers = select_sublayers(enumerate_sublayers(model), args.layers)
    sublayers.sort(key=lambda x: x[1], reverse=args.reverse)
    max_N = max(P // C + 1 for _, P in sublayers)
    print(f"\n{len(sublayers)} sublayers, max N = {max_N:,}")
    for name, P in sublayers:
        print(f"  {name:22s}  P={P:>7,}  N={P // C + 1:>6,}")

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

    for layer_name, P_layer in sublayers:
        npz_path = os.path.join(out_dir, f"{args.prefix}_{layer_name.replace('.', '_')}.npz")
        if os.path.exists(npz_path):
            print(f"\nSkipping {layer_name} (already done)")
            continue

        N_needed = P_layer // C + 1
        print(f"\n{'=' * 60}\n{layer_name} | P={P_layer:,} | N={N_needed:,}\n{'=' * 60}")
        model.load_state_dict(original_sd)

        t0 = time.time()
        jac = get_jac_layer(model, jac_loader, device, C, N_needed, layer_name)
        print(f"  Jacobian {jac.shape} in {time.time() - t0:.1f}s " f"({jac.nbytes / 1e9:.2f} GB)")

        # J is freed before the eigendecomposition: only G's eigenvectors are
        # needed downstream, and at P = 36,864 J is 10.9 GB on the device.
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
        del eigvecs, eigenvalues
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        layer_params, w0 = get_layer_params(model, layer_name)
        dtype = next(model.parameters()).dtype

        # With a complete basis the k = P projection is the identity on w0; a
        # nonzero residual means the eigensolver truncation was not repaired.
        recon_err = float(np.linalg.norm(U @ (U.T @ w0) - w0) / np.linalg.norm(w0))
        print(f"  Reconstruction error at k=P: {recon_err:.2e}")
        if recon_err > 1e-8:
            print(f"  WARNING: basis does not span R^P for {layer_name}")

        ks = k_schedule(P_layer, args.n_k_points, C)
        losses = {n: [] for n in SPLITS}
        accs = {n: [] for n in SPLITS}

        t0 = time.time()
        for k in tqdm(ks, desc=f"  {layer_name}"):
            proj = (U[:, :k] @ (U[:, :k].T @ w0)) if k <= U.shape[1] else w0
            set_layer_weights(layer_params, proj, dtype, device)
            refresh_bn_cached(model, bn_batches)
            for name in SPLITS:
                loss, acc = eval_cached(model, cached[name])
                losses[name].append(loss)
                accs[name].append(acc)
        print(f"  Sweep in {time.time() - t0:.1f}s")

        set_layer_weights(layer_params, w0, dtype, device)
        del U

        save = {
            "k_values": np.array(ks),
            "P": P_layer,
            "rank": rank,
            "n_repaired": n_repaired,
            "recon_err": recon_err,
            "fisher_set": args.fisher_set,
            "seed": args.seed,
            "layer": layer_name,
            "noise_fraction": exp["noise_fraction"],
            "n_jac_samples": len(jac_idx),
            "n_jac_corrupted": n_corrupt,
        }
        for name in SPLITS:
            save[name] = np.array(losses[name])
            save[name + "_acc"] = np.array(accs[name])
        np.savez(npz_path, **save)
        print(f"  Saved: {npz_path}")

    model.load_state_dict(original_sd)
    print("\nDone.")


if __name__ == "__main__":
    main()
