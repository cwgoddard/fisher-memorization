"""Predictive Fisher matrices and Conv/Linear weight projections."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader

# The standard divide-and-conquer symmetric eigensolver (LAPACK/cuSOLVER
# ``syevd``) needs a workspace of ~2*n^2 elements. Both the cuSOLVER path used
# by ``torch.linalg.eigh`` on CUDA and the 32-bit LAPACK interface used on CPU
# index that workspace with a signed 32-bit int, so once 2*n^2 exceeds
# INT32_MAX the buffer-size query fails with CUSOLVER_STATUS_INVALID_VALUE
# (see pytorch/pytorch#166004). This happens around n > 32768.
_INT32_MAX = 2**31 - 1


def _eigh_symmetric(fisher, device):
    """Symmetric eigendecomposition that scales to large matrices.

    Returns (evals, evecs) as CPU tensors with eigenvalues in ascending order
    (same convention as ``torch.linalg.eigh``).

    For matrices small enough that the ``syevd`` workspace fits in a 32-bit
    int, this uses the fast GPU ``torch.linalg.eigh``. For larger matrices it
    falls back to SciPy's ``evr`` (MRRR) driver, whose workspace is O(n) and
    therefore does not overflow.
    """
    if not torch.isfinite(fisher).all():
        n_bad = (~torch.isfinite(fisher)).sum().item()
        raise ValueError(
            f"Fisher matrix contains {n_bad} non-finite (NaN/Inf) entries; "
            "eigendecomposition would fail. Check the Fisher computation "
            "(e.g. exploding gradients) before decomposing."
        )

    n = fisher.shape[-1]
    needs_large_solver = 2 * n * n + 6 * n + 1 > _INT32_MAX

    if not needs_large_solver:
        evals, evecs = torch.linalg.eigh(fisher.to(device))
        return evals.cpu(), evecs.cpu()

    # Large matrix: the 32-bit syevd workspace would overflow, so use LAPACK's
    # MRRR driver via SciPy on CPU. Keep float32 to match the Fisher dtype and
    # halve the memory/time relative to float64.
    from scipy.linalg import eigh as scipy_eigh

    fisher_np = fisher.detach().cpu().contiguous().numpy()
    evals_np, evecs_np = scipy_eigh(fisher_np, driver="evr")
    return (
        torch.from_numpy(evals_np).to(fisher.dtype),
        torch.from_numpy(evecs_np).to(fisher.dtype),
    )


def _get_targeted_layers(model):
    """Collect Conv2d and Linear layers with depth metadata."""
    core = model.module if hasattr(model, "module") else model
    layers = []
    depth = 0
    for name, module in core.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append(
                {
                    "depth": depth,
                    "name": name,
                    "module": module,
                    "n_params": module.weight.numel(),
                }
            )
            depth += 1
    return layers


def _recalibrate_batchnorm(model, calibration_loader, device, max_batches=None, cumulative=True):
    """Reset and recompute BatchNorm running statistics via a calibration pass.

    After modifying model weights (e.g., by projection), the BatchNorm
    running mean/variance are stale. This function resets them and runs
    a forward pass over calibration data in train mode to recompute them. When
    requested, BatchNorm momentum is temporarily set to ``None`` so PyTorch
    uses a cumulative moving average and weights every calibration batch
    equally.
    """
    batchnorm_modules = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            batchnorm_modules.append((module, module.momentum))
            module.reset_running_stats()
            if cumulative:
                module.momentum = None

    try:
        model.train()
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(calibration_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                images = images.to(device, non_blocking=True)
                model(images)
    finally:
        for module, momentum in batchnorm_modules:
            module.momentum = momentum
        model.eval()


def layerwise_fisher_wrt_weights(model, data_loader, device, max_batches=None, target_depths=None):
    """Compute the predictive Fisher for selected Conv/Linear weights.

    F_l = (1/N) sum_x sum_c p(c|x) grad_w log p(c|x) grad_w log p(c|x)^T.
    Returns CPU Fisher matrices and layer names indexed by depth.
    """
    model.eval()
    core = model.module if hasattr(model, "module") else model
    layers = _get_targeted_layers(model)
    if target_depths is not None:
        layers = [e for e in layers if e["depth"] in target_depths]

    if len(layers) == 0:
        raise ValueError("No Conv/Linear layers found.")

    weight_tensors = [entry["module"].weight for entry in layers]
    fisher_by_depth = {
        entry["depth"]: torch.zeros(entry["n_params"], entry["n_params"], device=device)
        for entry in layers
    }

    n_samples = 0

    for batch_idx, (images, _) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        logits = model(images)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp().detach()

        bsz, n_classes = logits.shape

        for b in range(bsz):
            n_samples += 1
            for c in range(n_classes):
                core.zero_grad(set_to_none=True)

                grads_w = torch.autograd.grad(
                    log_probs[b, c],
                    weight_tensors,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )

                p_bc = probs[b, c]

                for entry, g_w in zip(layers, grads_w):
                    g_flat = g_w.detach().reshape(-1)
                    fisher_by_depth[entry["depth"]] += p_bc * torch.outer(g_flat, g_flat)

    if n_samples == 0:
        raise ValueError("No samples processed. Check data_loader/max_batches.")

    fisher_out = {depth: (mat / n_samples).detach().cpu() for depth, mat in fisher_by_depth.items()}
    layer_name_map = {entry["depth"]: entry["name"] for entry in layers}
    return fisher_out, layer_name_map


class LayerwiseFisherProjector:
    """Cache layerwise Fisher eigenbases for weight projection and removal."""

    def __init__(
        self,
        model,
        dataset,
        device,
        layer_scope="first",
        fisher_batch_size=None,
        fisher_num_workers=0,
        max_fisher_batches=None,
        fisher_loader=None,
    ):
        self.device = device
        self.fisher_batch_size = fisher_batch_size
        self.model = model

        all_layers = _get_targeted_layers(model)
        if len(all_layers) == 0:
            raise ValueError("No Conv/Linear layers found.")

        all_depths = sorted(entry["depth"] for entry in all_layers)
        if layer_scope == "all":
            self.selected_depths = set(all_depths)
        elif layer_scope == "first":
            self.selected_depths = {all_depths[0]}
        elif layer_scope == "last":
            self.selected_depths = {all_depths[-1]}
        elif isinstance(layer_scope, int):
            if layer_scope < 0 or layer_scope >= len(all_depths):
                raise ValueError(
                    f"layer_scope index {layer_scope} out of range " f"[0, {len(all_depths) - 1}]."
                )
            self.selected_depths = {all_depths[layer_scope]}
        elif isinstance(layer_scope, str):
            matching = [e for e in all_layers if e["name"] == layer_scope]
            if not matching:
                available = [e["name"] for e in all_layers]
                raise ValueError(
                    f"layer_scope '{layer_scope}' is not a recognized keyword "
                    f"('all', 'first', 'last') or layer name. "
                    f"Available names: {available}"
                )
            self.selected_depths = {matching[0]["depth"]}
        else:
            raise ValueError(
                "layer_scope must be 'all', 'first', 'last', an int index, "
                "or a layer name string."
            )

        selected_layers = [e for e in all_layers if e["depth"] in self.selected_depths]
        self.layer_name_map = {e["depth"]: e["name"] for e in selected_layers}

        if fisher_loader is None:
            if isinstance(dataset, DataLoader):
                fisher_loader = dataset
            else:
                fisher_loader = DataLoader(
                    dataset,
                    batch_size=fisher_batch_size,
                    shuffle=False,
                    num_workers=fisher_num_workers,
                    pin_memory=True,
                )
        self.fisher_loader = fisher_loader

        self.fisher_by_depth, _ = layerwise_fisher_wrt_weights(
            model=self.model,
            data_loader=self.fisher_loader,
            device=device,
            max_batches=max_fisher_batches,
            target_depths=self.selected_depths,
        )

        # Eigendecompose each Fisher once and cache the results.
        # eigh returns eigenvalues in ascending order; the top-k eigenvectors
        # are the last k columns of evecs.
        self.eigen_by_depth = {}
        for depth, fisher in self.fisher_by_depth.items():
            evals, evecs = _eigh_symmetric(fisher, device)
            self.eigen_by_depth[depth] = (evals, evecs)


# Full-model single-sample Fisher with rank trick


def full_model_single_sample_fisher_top_k(model, image, device, k=9):
    """Top single-sample Fisher eigenpairs across Conv/Linear weights.

    For C softmax classes the Fisher has rank at most C-1. Build the
    probability-weighted gradient factor G (P x C), diagonalize G.T @ G,
    then lift and normalize the requested directions in parameter space.
    Labels are unused. Returns descending eigenvalues and vectors on CPU.
    """
    model.eval()
    layers = _get_targeted_layers(model)
    weight_params = [entry["module"].weight for entry in layers]

    x = image.unsqueeze(0).to(device)
    logits = model(x)
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp().detach()
    n_classes = logits.shape[1]

    scaled_grads = []
    for c in range(n_classes):
        model.zero_grad()
        grads = torch.autograd.grad(
            log_probs[0, c], weight_params, retain_graph=True, create_graph=False
        )
        g_flat = torch.cat([g.detach().reshape(-1) for g in grads])
        scaled_grads.append(torch.sqrt(probs[0, c]) * g_flat)

    G = torch.stack(scaled_grads, dim=1)  # (P, C)
    small_mat = G.T @ G  # (C, C)
    small_evals, small_evecs = torch.linalg.eigh(small_mat)  # ascending

    k_eff = min(k, n_classes)
    top_small_evals = small_evals[-k_eff:]
    top_small_evecs = small_evecs[:, -k_eff:]

    top_evecs_param = G @ top_small_evecs  # (P, k_eff)
    norms = top_evecs_param.norm(dim=0, keepdim=True).clamp(min=1e-30)
    top_evecs_param = top_evecs_param / norms

    return top_small_evals.flip(0).cpu(), top_evecs_param.flip(1).cpu()


def flatten_all_weights(model_or_layers):
    """Concatenate all Conv/Linear .weight tensors into a single flat vector."""
    if not isinstance(model_or_layers, list):
        model_or_layers = _get_targeted_layers(model_or_layers)
    return torch.cat([e["module"].weight.data.reshape(-1) for e in model_or_layers])


def scatter_flat_weights(flat_vector, layers):
    """Copy a flat parameter vector back into per-layer .weight tensors.

    Inverse of flatten_all_weights: partitions ``flat_vector`` according
    to each layer's parameter count and writes back into the corresponding
    weight tensors (in-place).
    """
    offset = 0
    for entry in layers:
        n = entry["n_params"]
        entry["module"].weight.data.copy_(
            flat_vector[offset : offset + n].view_as(entry["module"].weight)
        )
        offset += n
