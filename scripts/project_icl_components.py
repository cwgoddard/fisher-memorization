#!/usr/bin/env python3
"""Run Fisher-only component analysis with weight-space ensembling."""

import argparse
import json
import os
import time

import numpy as np
import torch

import _bootstrap  # noqa: F401
from src import icl_data as dataset_utils
from src import transformer
from src.icl_eval import test_model


def get_batch(ws, pretrain_size, batch_size):
    if pretrain_size >= batch_size:
        idxs = np.random.permutation(pretrain_size)[:batch_size]
    else:
        n_full = batch_size // pretrain_size
        idxs = np.tile(np.arange(pretrain_size), n_full)
        remainder = batch_size - len(idxs)
        if remainder:
            idxs = np.concatenate([idxs, np.random.permutation(pretrain_size)[:remainder]])
        np.random.shuffle(idxs)
    return ws[idxs]


def flatten_params(params):
    return torch.cat([p.data.flatten() for p in params])


def load_flat_params(params, flat):
    ptr = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[ptr : ptr + n].reshape(p.shape))
        ptr += n


def _parse_block_indices(arg: str | None, n_blocks: int) -> list[int]:
    if arg is None or str(arg).strip() == "":
        return list(range(n_blocks))
    out = []
    for tok in str(arg).split(","):
        tok = tok.strip()
        if not tok:
            continue
        i = int(tok)
        if i < 0 or i >= n_blocks:
            raise ValueError(f"Block index {i} out of range [0, {n_blocks - 1}]")
        out.append(i)
    if not out:
        raise ValueError("No valid block indices provided.")
    return sorted(set(out))


def make_groups(model, block_indices=None):
    groups = []
    n_blocks = len(model.transformer.blocks)
    keep = _parse_block_indices(block_indices, n_blocks)
    for block_idx in keep:
        block = model.transformer.blocks[block_idx]
        attn_params = list(block.sa.parameters()) + list(block.ln1.parameters())
        mlp_params = list(block.ffwd.parameters()) + list(block.ln2.parameters())
        groups.append((f"Block {block_idx} Attention", f"block{block_idx}_attn", attn_params))
        groups.append((f"Block {block_idx} MLP", f"block{block_idx}_mlp", mlp_params))
    return groups


def run_component_fisher_wse(model, ws, ws_tiled, criterion, grad_idxs, device, hp, seed_dir, args):
    groups = make_groups(model, block_indices=args.block_indices)

    model.eval()
    with torch.no_grad():
        ref_train = test_model(
            model,
            device,
            grad_idxs,
            criterion,
            1,
            hp["dim"],
            hp["batch_size"],
            1,
            hp["seq_len"],
            hp["noise_std"],
            offset=500_000,
            ws=ws_tiled,
            test_batches=hp["test_batches"],
            lastonly=True,
        )
        ref_test = test_model(
            model,
            device,
            grad_idxs,
            criterion,
            1,
            hp["dim"],
            hp["batch_size"],
            1,
            hp["seq_len"],
            hp["noise_std"],
            offset=600_000,
            ws=None,
            norm=True,
            test_batches=hp["test_batches"],
            lastonly=True,
        )

    print(
        f"\nComponent Fisher WSE (alpha={args.ensemble_alpha:.3f}) — "
        f"full model refs  train: {ref_train:.4f}  held-out: {ref_test:.4f}\n",
        flush=True,
    )

    out_npz = {}
    target_device = torch.device(device)
    if args.fisher_eigh_device == "cpu":
        eigh_device = torch.device("cpu")
    else:
        if target_device.type != "cuda":
            raise ValueError("--fisher-eigh-device=cuda requested, but CUDA is unavailable.")
        eigh_device = target_device

    alpha = float(args.ensemble_alpha)
    for group_idx, (label, slug, params) in enumerate(groups):
        n_params = sum(p.numel() for p in params)
        original = flatten_params(params).clone()
        n_fisher = n_params

        print(
            f"{label} ({n_params:,} params): collecting {n_fisher:,} gradients for Fisher …",
            flush=True,
        )
        t_fisher = time.time()
        F = torch.zeros(n_params, n_params, device=device)
        for i in range(n_fisher):
            model.zero_grad()
            model.train()
            batch_ws = get_batch(ws, hp["pretrain_size"], hp["batch_size"])
            xs, ys, _ = dataset_utils.gen_linreg_data(
                hp["total_steps"] + group_idx * n_fisher + i,
                batch_size=hp["batch_size"],
                dim=hp["dim"],
                n_samples=hp["seq_len"],
                device=device,
                noise_std=hp["noise_std"],
                ws=batch_ws,
            )
            pred = model((xs, ys))[:, grad_idxs].squeeze()
            true_ys = ys[:, :, 0].squeeze()
            loss = criterion(pred, true_ys)
            loss.backward()
            g = torch.cat([p.grad.flatten() for p in params]).detach()
            F.addmm_(g.unsqueeze(1), g.unsqueeze(0), alpha=1.0, beta=1.0)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{n_fisher} gradients ({time.time() - t_fisher:.0f}s)", flush=True)

        F /= n_fisher
        print(
            f"  Gradients done ({time.time() - t_fisher:.0f}s). "
            f"Running eigh on {eigh_device.type} …",
            flush=True,
        )
        F_eigh = F.cpu() if eigh_device.type == "cpu" else F
        del F
        eigenvalues, eigenvectors = torch.linalg.eigh(F_eigh)
        del F_eigh
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
        if eigenvectors.device.type != target_device.type:
            eigenvalues = eigenvalues.to(target_device)
            eigenvectors = eigenvectors.to(target_device)

        k_step = args.k_step if args.k_step is not None else hp["k_step"]
        ks = [1] + list(range(k_step, n_params, k_step))
        if ks[-1] != n_params:
            ks.append(n_params)

        train_losses, test_losses = [], []
        for k in ks:
            V_k = eigenvectors[:, :k]
            projected = V_k @ (V_k.T @ original)
            blended = (1.0 - alpha) * original + alpha * projected
            load_flat_params(params, blended)

            model.eval()
            with torch.no_grad():
                tl = test_model(
                    model,
                    device,
                    grad_idxs,
                    criterion,
                    1,
                    hp["dim"],
                    hp["batch_size"],
                    1,
                    hp["seq_len"],
                    hp["noise_std"],
                    offset=500_000,
                    ws=ws_tiled,
                    test_batches=hp["test_batches"],
                    lastonly=True,
                )
                hl = test_model(
                    model,
                    device,
                    grad_idxs,
                    criterion,
                    1,
                    hp["dim"],
                    hp["batch_size"],
                    1,
                    hp["seq_len"],
                    hp["noise_std"],
                    offset=600_000,
                    ws=None,
                    norm=True,
                    test_batches=hp["test_batches"],
                    lastonly=True,
                )
            train_losses.append(tl)
            test_losses.append(hl)
            if k == 1 or k % (k_step * 5) == 0 or k == n_params:
                print(f"  k={k:5d}/{n_params}  train={tl:.4f}  held_out={hl:.4f}", flush=True)

        del eigenvalues, eigenvectors
        load_flat_params(params, original)

        out_npz[f"{slug}_ks"] = np.array(ks)
        out_npz[f"{slug}_train_losses"] = np.array(train_losses)
        out_npz[f"{slug}_test_losses"] = np.array(test_losses)

    out_npz["_group_labels"] = np.array([g[0] for g in groups], dtype=object)
    out_npz["_group_slugs"] = np.array([g[1] for g in groups], dtype=object)
    out_npz["_ref_train"] = np.array(ref_train, dtype=np.float64)
    out_npz["_ref_test"] = np.array(ref_test, dtype=np.float64)
    out_npz["_ensemble_alpha"] = np.array(alpha, dtype=np.float64)

    npz_path = os.path.join(seed_dir, f"{args.output_name}.npz")
    np.savez(npz_path, **out_npz)

    print(f"Wrote {npz_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run Fisher-only component analysis with weight-space ensembling."
    )
    parser.add_argument("--results-dir", type=str, default="results/icl")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, default="model_final.pt")
    parser.add_argument("--ensemble-alpha", type=float, default=0.5)
    parser.add_argument(
        "--block-indices",
        type=str,
        default=None,
        help="Comma-separated transformer block indices to analyze (default: all). Example: 1",
    )
    parser.add_argument(
        "--fisher-eigh-device",
        type=str,
        choices=("cpu", "cuda"),
        default="cuda",
        help="Device for Fisher eigensolve.",
    )
    parser.add_argument(
        "--k-step",
        type=int,
        default=None,
        help="Override k-step for projection sweep (default: from run_config.json).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="fisher_component_projection_wse",
        help="Prefix for the output npz inside the seed directory.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.ensemble_alpha <= 1.0):
        raise ValueError("--ensemble-alpha must be in [0, 1].")

    seed_dir = os.path.join(args.results_dir, f"seed_{args.seed}")
    cfg_path = os.path.join(seed_dir, "run_config.json")
    ckpt_path = os.path.join(seed_dir, args.checkpoint)
    ws_path = os.path.join(seed_dir, "ws_train.pt")
    for p in [cfg_path, ckpt_path, ws_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required file: {p}")
    with open(cfg_path, "r") as f:
        hp = json.load(f)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Seed: {args.seed}  Device: {device}  Input: {seed_dir}", flush=True)
    if args.fisher_eigh_device == "cuda" and device != "cuda":
        raise ValueError("--fisher-eigh-device=cuda requested, but CUDA is unavailable.")

    if hp["d_model"] % hp["n_head"] != 0:
        raise ValueError(f"d_model ({hp['d_model']}) must be divisible by n_head ({hp['n_head']})")

    model = transformer.RegressionTransformer(
        d_model=hp["d_model"],
        device=device,
        block_size=hp["seq_len"] * 2,
        n_layer=hp["n_layer"],
        input_dim=hp["dim"],
        n_head=hp["n_head"],
        dropout=hp["dropout"],
        expansion_factor=hp["ff_factor"],
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ws = torch.load(ws_path, map_location=device)
    reps = int(np.ceil(hp["batch_size"] / hp["pretrain_size"]))
    ws_tiled = ws.repeat(reps, 1)[: hp["batch_size"]]
    criterion = torch.nn.MSELoss()
    grad_idxs = np.arange(0, 2 * hp["seq_len"], 2)

    run_component_fisher_wse(
        model=model,
        ws=ws,
        ws_tiled=ws_tiled,
        criterion=criterion,
        grad_idxs=grad_idxs,
        device=device,
        hp=hp,
        seed_dir=seed_dir,
        args=args,
    )


if __name__ == "__main__":
    main()
