"""Fisher information: Jacobians, eigenbases, and projections."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.func import functional_call, jacrev, vmap

from .resnet import resnet20

# Mixed into the Fisher-sampling seed so it draws on a stream independent of
# the label-noise draw. See draw_fisher_samples.
FISHER_DRAW_TAG = 0xF15E5


def load_resnet20(path, device="cuda"):
    """Load a checkpoint saved by train_label_noise.py, with or without a DataParallel wrap."""
    state = torch.load(path, map_location=device, weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = resnet20().to(device)
    model.load_state_dict(state)
    return model


def draw_fisher_samples(pool, n, seed, is_noised=None, sort=False):
    """Draw Fisher indices using an RNG stream independent of label corruption."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, FISHER_DRAW_TAG]))
    idx = rng.choice(pool, size=min(n, len(pool)), replace=False)
    if sort:
        idx.sort()
    if is_noised is not None:
        check_draw_independence(idx, pool, is_noised)
    return idx


def check_draw_independence(idx, pool, is_noised):
    """Warn if the Fisher sample's corrupted fraction departs from the pool's."""
    f_pop = float(is_noised[pool].mean())
    f_smp = float(is_noised[idx].mean())
    sd = float(np.sqrt(max(f_pop * (1.0 - f_pop), 1e-12) / len(idx)))
    if abs(f_smp - f_pop) > 6 * sd + 1e-9:
        print(
            f"WARNING: sampled corrupted fraction {f_smp:.4f} is "
            f"{abs(f_smp - f_pop) / max(sd, 1e-12):.0f} sigma from the pool's "
            f"{f_pop:.4f} -- is the Fisher draw correlated with the "
            f"label-noise draw?",
            flush=True,
        )


def cache_split(dataset, indices, n_batches, bs, device):
    """Move up to n_batches*bs samples onto the device as fixed batches.

    The sweeps re-evaluate the same splits at every one of hundreds of k
    values, which makes a DataLoader-per-k image-decode bound. Equivalent to
    iterating a shuffle=False loader over Subset(dataset, indices) and stopping
    after n_batches, because load_cifar10's transform is deterministic.
    """
    take = np.asarray(indices)[: n_batches * bs]
    n_workers = min(4, max(1, len(os.sched_getaffinity(0)) - 1))
    loader = tud.DataLoader(
        tud.Subset(dataset, take.tolist()),
        batch_size=bs,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )
    return [(x.to(device, non_blocking=True), y.to(device, non_blocking=True)) for x, y in loader]


def cache_eval_splits(splits, n_batches, bs, device, order=None):
    """Cache a {name: (dataset, indices)} mapping, optionally reordering indices."""
    out = {}
    for name, (ds, idx) in splits.items():
        idx = np.arange(len(ds)) if idx is None else idx
        out[name] = cache_split(ds, order(idx) if order else idx, n_batches, bs, device)
    return out


def refresh_bn_cached(model, batches):
    """Reset BatchNorm statistics and recompute them by cumulative average."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None
    model.train()
    with torch.no_grad():
        for x, _ in batches:
            model(x)


def eval_cached(model, batches):
    """Loss and accuracy over pre-cached batches."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in batches:
            out = model(x)
            total_loss += F.cross_entropy(out, y, reduction="sum").item()
            correct += out.argmax(1).eq(y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def get_jac_layer(model, loader, device, C, N, layer_name):
    """Fisher-scaled Jacobian of one sublayer, shape (P, N*C).

    Column i*C + c holds d[2*sqrt(softmax(f(x_i))_c)]/dtheta, so J J^T is the
    Fisher F = (1/N) sum_i sum_c p_ic s_ic s_ic^T. The targets are never read:
    F is an expectation over the model's own predictive distribution.

    jacrev runs in the model's dtype (float32); the result is accumulated and
    returned as float64, which is what the eigendecompositions need.
    """
    model.eval()
    base = model.module if hasattr(model, "module") else model

    prefix = layer_name + "."
    all_params = dict(base.named_parameters())
    layer_param_names = [n for n in all_params if n.startswith(prefix)]
    if not layer_param_names:
        raise ValueError(f"No params for '{layer_name}'")

    layer_params = {n: all_params[n] for n in layer_param_names}
    other_params = {n: p.detach() for n, p in all_params.items() if n not in layer_param_names}
    buffers = dict(base.named_buffers())
    P = sum(p.numel() for p in layer_params.values())

    def f_single(lp, op, x):
        merged = {**op, **lp}
        out = functional_call(base, (merged, buffers), (x.unsqueeze(0),))
        return 2.0 * torch.sqrt(F.softmax(out, dim=1).squeeze(0))

    jac_fn = vmap(jacrev(f_single, argnums=0), in_dims=(None, None, 0))

    jac = np.empty((N * C, P), dtype=np.float64)
    row, n_samples = 0, 0
    for data, _ in loader:
        if n_samples >= N:
            break
        B = min(data.size(0), N - n_samples)
        data = data[:B].to(device)
        jac_dict = jac_fn(layer_params, other_params, data)
        flat = [jac_dict[n].reshape(B, C, -1) for n in layer_param_names]
        jac[row : row + B * C] = torch.cat(flat, dim=-1).reshape(B * C, P).detach().cpu().numpy()
        row += B * C
        n_samples += B

    return (jac / np.sqrt(N)).T


def descending_eigh(G):
    """Eigendecompose a Gram matrix, ordered by decreasing eigenvalue.

    Returns (eigenvalues, U, rank), where rank counts eigenvalues above
    lambda_max * 1e-10. Callers upload and free the Jacobian themselves: at
    P = 36,864 it is 10.9 GB on the device, so whether it is still alive
    during the eigendecomposition decides whether the job fits.
    """
    eigenvalues, U = torch.linalg.eigh(G)
    eigenvalues = eigenvalues.flip(0).contiguous()
    U = U.flip(1).contiguous()
    rank = int((eigenvalues > float(eigenvalues[0]) * 1e-10).sum().item())
    return eigenvalues, U, rank


def complete_basis(U, device, tol=0.5):
    """Rebuild the zero columns ``torch.linalg.eigh`` returns past 2^15 on CUDA.

    Without this, U U^T projects onto the recovered block rather than the
    identity, so a sweep never returns to the trained weights at k = P. The
    missing columns sit in the numerically null block, where any orthonormal
    completion is an equally valid eigenbasis. Valid columns are moved first,
    keeping their relative order. See docs/numerics.md.
    """
    P = U.shape[0]
    good = U.norm(dim=0) > tol
    n_good = int(good.sum().item())
    n_bad = P - n_good
    if n_bad == 0:
        return U, 0

    if not bool(good[:n_good].all().item()):
        order = torch.argsort((~good).to(torch.int8), stable=True)
        U = U[:, order].contiguous()

    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(0)
    M = torch.randn(P, n_bad, generator=gen, device=device, dtype=U.dtype)
    Q_good = U[:, :n_good]
    for _ in range(2):  # repeated projection for numerical stability
        M -= Q_good @ (Q_good.T @ M)
    Qn, _ = torch.linalg.qr(M)
    del M
    U[:, n_good:] = Qn
    del Qn
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    print(
        f"  Repaired {n_bad:,} defective eigenvector columns " f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    return U, n_bad


def enumerate_sublayers(base_model):
    """(name, P) for every conv/linear sublayer, in module order."""
    return [
        (name, sum(p.numel() for p in mod.parameters()))
        for name, mod in base_model.named_modules()
        if isinstance(mod, (nn.Conv2d, nn.Linear))
    ]


def select_sublayers(sublayers, spec):
    """Filter (name, P) pairs by a comma-separated list of names."""
    if spec is None:
        return sublayers
    wanted = {n.strip() for n in spec.split(",") if n.strip()}
    chosen = [(n, p) for n, p in sublayers if n in wanted]
    if not chosen:
        raise SystemExit(f"No sublayers matched {spec!r}")
    return chosen


def get_layer_params(base_model, layer_name):
    """[(name, param)] for a sublayer, and its flattened weight vector."""
    prefix = layer_name + "."
    layer_params = [(n, p) for n, p in base_model.named_parameters() if n.startswith(prefix)]
    w0 = torch.cat([p.data.flatten() for _, p in layer_params]).cpu().numpy().copy()
    return layer_params, w0


def set_layer_weights(layer_params, proj, dtype, device):
    """Write a flat vector back into a sublayer's parameter tensors."""
    idx = 0
    for _, p in layer_params:
        n = p.numel()
        p.data = torch.from_numpy(proj[idx : idx + n].reshape(p.size())).to(
            dtype=dtype, device=device
        )
        idx += n


def k_schedule(P, n_points, C=10):
    """Log-spaced k values, with every k <= C included and k = P at the end."""
    ks = set(range(1, min(C, P) + 1))
    ks.update(np.unique(np.geomspace(1, P, n_points).astype(int)).tolist())
    ks.add(P)
    return sorted(ks)
