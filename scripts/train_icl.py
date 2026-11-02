#!/usr/bin/env python3
"""Transient ICL generalization experiment — standalone script for cluster use."""

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


def save_run_config_json(out_dir, args_ns):
    path = os.path.join(out_dir, "run_config.json")
    blob = vars(args_ns).copy()
    for k, v in list(blob.items()):
        if hasattr(v, "item"):
            blob[k] = v.item()
    with open(path, "w") as fp:
        json.dump(blob, fp, indent=2, sort_keys=True)


def train(device, out_dir, hp):
    angle_rad = hp.angle * np.pi / 180
    ws = dataset_utils.sample_cone(hp.pretrain_size, hp.dim, angle_rad, device=device)
    torch.save(ws, os.path.join(out_dir, "ws_train.pt"))

    if hp.d_model % hp.n_head != 0:
        raise ValueError(f"d-model ({hp.d_model}) must be divisible by n-head ({hp.n_head})")

    model = transformer.RegressionTransformer(
        d_model=hp.d_model,
        device=device,
        block_size=hp.seq_len * 2,
        n_layer=hp.n_layer,
        input_dim=hp.dim,
        n_head=hp.n_head,
        dropout=hp.dropout,
        expansion_factor=hp.ff_factor,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    criterion = torch.nn.MSELoss()
    grad_idxs = np.arange(0, 2 * hp.seq_len, 2)

    reps = int(np.ceil(hp.batch_size / hp.pretrain_size))
    ws_tiled = ws.repeat(reps, 1)[: hp.batch_size]

    log_steps = []
    train_task_loss_history = []
    held_out_loss_history = []
    best_held_out = float("inf")

    t0 = time.time()
    for step in range(hp.total_steps):
        model.train()
        batch_ws = get_batch(ws, hp.pretrain_size, hp.batch_size)
        xs, ys, _ = dataset_utils.gen_linreg_data(
            step,
            batch_size=hp.batch_size,
            dim=hp.dim,
            n_samples=hp.seq_len,
            device=device,
            noise_std=hp.noise_std,
            ws=batch_ws,
        )
        optimizer.zero_grad()
        pred = model((xs, ys))[:, grad_idxs].squeeze()
        true_ys = ys[:, :, 0].squeeze()
        loss = criterion(pred, true_ys)
        loss.backward()
        optimizer.step()

        if step % hp.eval_every == 0:
            model.eval()
            with torch.no_grad():
                train_loss = test_model(
                    model,
                    device,
                    grad_idxs,
                    criterion,
                    step + 1,
                    hp.dim,
                    hp.batch_size,
                    1,
                    hp.seq_len,
                    hp.noise_std,
                    offset=100_000,
                    ws=ws_tiled,
                    test_batches=hp.test_batches,
                    lastonly=True,
                )
                held_out_loss = test_model(
                    model,
                    device,
                    grad_idxs,
                    criterion,
                    step + 1,
                    hp.dim,
                    hp.batch_size,
                    1,
                    hp.seq_len,
                    hp.noise_std,
                    offset=200_000,
                    ws=None,
                    norm=True,
                    test_batches=hp.test_batches,
                    lastonly=True,
                )
            log_steps.append(step)
            train_task_loss_history.append(train_loss)
            held_out_loss_history.append(held_out_loss)

            if held_out_loss < best_held_out:
                best_held_out = held_out_loss
                torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))

            if step % 2000 == 0:
                elapsed = time.time() - t0
                print(
                    f"Step {step:6d}/{hp.total_steps}  "
                    f"train_task={train_loss:.4f}  "
                    f"held_out={held_out_loss:.4f}  "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )

    log_steps = np.array(log_steps)
    train_task_loss_history = np.array(train_task_loss_history)
    held_out_loss_history = np.array(held_out_loss_history)

    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pt"))
    np.savez(
        os.path.join(out_dir, "training_curves.npz"),
        log_steps=log_steps,
        train_task_loss_history=train_task_loss_history,
        held_out_loss_history=held_out_loss_history,
    )

    gap = held_out_loss_history[-1] - held_out_loss_history.min()

    best_idx = np.argmin(held_out_loss_history)
    best_step = log_steps[best_idx]
    best_val = held_out_loss_history[best_idx]

    print(f"\nBest held-out loss {best_val:.4f} at step {best_step}", flush=True)
    print(f"Final held-out loss {held_out_loss_history[-1]:.4f}", flush=True)
    print(
        f"Transient gap (final − best held-out): {gap:.4f} "
        f"(positive ⇒ recovery / specialization after peak)",
        flush=True,
    )
    print(f"Final train-task loss {train_task_loss_history[-1]:.4f}", flush=True)

    return model, ws, ws_tiled, criterion, grad_idxs


def fisher_analysis(model, ws, ws_tiled, criterion, grad_idxs, device, out_dir, hp):
    blocks = list(model.transformer.blocks)

    model.eval()
    with torch.no_grad():
        ref_train = test_model(
            model,
            device,
            grad_idxs,
            criterion,
            1,
            hp.dim,
            hp.batch_size,
            1,
            hp.seq_len,
            hp.noise_std,
            offset=500_000,
            ws=ws_tiled,
            test_batches=hp.test_batches,
            lastonly=True,
        )
        ref_test = test_model(
            model,
            device,
            grad_idxs,
            criterion,
            1,
            hp.dim,
            hp.batch_size,
            1,
            hp.seq_len,
            hp.noise_std,
            offset=600_000,
            ws=None,
            norm=True,
            test_batches=hp.test_batches,
            lastonly=True,
        )
    print(
        f"\nFisher analysis — full model  train: {ref_train:.4f}  held-out: {ref_test:.4f}\n",
        flush=True,
    )

    for block_idx, block in enumerate(blocks):
        params = list(block.parameters())
        n_params = sum(p.numel() for p in params)
        original = flatten_params(params).clone()

        n_fisher = n_params
        print(
            f"Block {block_idx}  ({n_params:,} params): "
            f"collecting {n_fisher:,} gradients for full-rank Fisher …",
            flush=True,
        )

        F = torch.zeros(n_params, n_params, device=device)
        t_fisher = time.time()
        for i in range(n_fisher):
            model.zero_grad()
            model.train()
            batch_ws = get_batch(ws, hp.pretrain_size, hp.batch_size)
            xs, ys, _ = dataset_utils.gen_linreg_data(
                hp.total_steps + block_idx * n_fisher + i,
                batch_size=hp.batch_size,
                dim=hp.dim,
                n_samples=hp.seq_len,
                device=device,
                noise_std=hp.noise_std,
                ws=batch_ws,
            )
            pred = model((xs, ys))[:, grad_idxs].squeeze()
            true_ys = ys[:, :, 0].squeeze()
            loss = criterion(pred, true_ys)
            loss.backward()
            g = torch.cat([p.grad.flatten() for p in params]).detach()
            F.addmm_(g.unsqueeze(1), g.unsqueeze(0), alpha=1.0, beta=1.0)
            if (i + 1) % 2000 == 0:
                print(
                    f"    {i+1}/{n_fisher} gradients  ({time.time()-t_fisher:.0f}s)",
                    flush=True,
                )
        F /= n_fisher
        target_device = torch.device(device)
        if hp.fisher_eigh_device == "cpu":
            eigh_device = torch.device("cpu")
        else:
            if target_device.type != "cuda":
                raise ValueError(
                    "--fisher-eigh-device=cuda requested, but training device is not CUDA."
                )
            eigh_device = target_device
        print(
            f"  Gradients done ({time.time()-t_fisher:.0f}s). "
            f"Running eigh on {eigh_device.type} …",
            flush=True,
        )

        if eigh_device.type == "cpu":
            F_eigh = F.cpu()
            del F
        else:
            F_eigh = F
            del F
        eigenvalues, eigenvectors = torch.linalg.eigh(F_eigh)
        del F_eigh
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
        if eigenvectors.device.type != target_device.type:
            eigenvalues = eigenvalues.to(target_device)
            eigenvectors = eigenvectors.to(target_device)

        print(f"  eigh done ({time.time()-t_fisher:.0f}s). Sweeping k …", flush=True)

        ks = [1] + list(range(hp.k_step, n_params, hp.k_step))
        if ks[-1] != n_params:
            ks.append(n_params)

        train_losses, test_losses = [], []
        for k in ks:
            V_k = eigenvectors[:, :k]
            projected = V_k @ (V_k.T @ original)
            load_flat_params(params, projected)

            model.eval()
            with torch.no_grad():
                tl = test_model(
                    model,
                    device,
                    grad_idxs,
                    criterion,
                    1,
                    hp.dim,
                    hp.batch_size,
                    1,
                    hp.seq_len,
                    hp.noise_std,
                    offset=500_000,
                    ws=ws_tiled,
                    test_batches=hp.test_batches,
                    lastonly=True,
                )
                hl = test_model(
                    model,
                    device,
                    grad_idxs,
                    criterion,
                    1,
                    hp.dim,
                    hp.batch_size,
                    1,
                    hp.seq_len,
                    hp.noise_std,
                    offset=600_000,
                    ws=None,
                    norm=True,
                    test_batches=hp.test_batches,
                    lastonly=True,
                )
            train_losses.append(tl)
            test_losses.append(hl)
            if k == 1 or k % (hp.k_step * 5) == 0 or k == n_params:
                print(
                    f"  k={k:5d}/{n_params}  train={tl:.4f}  held_out={hl:.4f}",
                    flush=True,
                )

        del eigenvectors, eigenvalues
        load_flat_params(params, original)

        np.savez(
            os.path.join(out_dir, f"fisher_data_block{block_idx}.npz"),
            ks=np.array(ks),
            train_losses=np.array(train_losses),
            test_losses=np.array(test_losses),
            ref_train=np.float64(ref_train),
            ref_test=np.float64(ref_test),
        )

    print("Fisher analysis complete.", flush=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Transient ICL + Fisher projection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/icl",
        help="Base directory; checkpoints go to OUTPUT_DIR/seed_<seed>/",
    )
    parser.add_argument(
        "--skip-fisher",
        action="store_true",
        help="Train and write training_curves.npz only (fast screening)",
    )
    # --- task / sampling ---
    parser.add_argument("--angle", type=float, default=180)
    parser.add_argument("--pretrain-size", type=int, default=2**4)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.0)
    # --- training ---
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--total-steps", type=int, default=120_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--test-batches", type=int, default=5)
    # --- model ---
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--ff-factor", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    # --- fisher sweep resolution ---
    parser.add_argument("--k-step", type=int, default=100)
    parser.add_argument(
        "--fisher-eigh-device",
        type=str,
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for Fisher eigensolve; CPU avoids CUDA OOM for larger blocks.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    save_run_config_json(out_dir, args)

    print(f"Seed: {args.seed}  Device: {device}  Output: {out_dir}", flush=True)

    model, ws, ws_tiled, criterion, grad_idxs = train(device, out_dir, args)
    if not args.skip_fisher:
        fisher_analysis(model, ws, ws_tiled, criterion, grad_idxs, device, out_dir, args)


if __name__ == "__main__":
    main()
