"""Held-out evaluation for in-context regression models."""

from . import icl_data as dataset_utils


def test_model(
    model,
    device,
    grad_idxs,
    criterion,
    epoch,
    dim,
    batch_size,
    batches_per_epoch,
    seq_len,
    noise_std,
    offset=1000,
    test_batches=50,
    ws=None,
    norm=False,
    lastonly=False,
):
    """Test the model on a set of weight vectors.
    If ws is None, sample from the full sphere.
    """
    total_loss = 0
    for b in range(test_batches):
        model.eval()
        # offset to ensure we don't test on the same data as training
        batch_seed = b + offset * batches_per_epoch * epoch
        xs, ys, ws = dataset_utils.gen_linreg_data(
            batch_seed,
            batch_size=batch_size,
            dim=dim,
            n_samples=seq_len,
            device=device,
            noise_std=noise_std,
            ws=ws,
            norm=norm,
        )

        outputs = model((xs, ys))
        pred = outputs[:, grad_idxs].squeeze()
        true_ys = ys[:, :, 0].squeeze()
        if lastonly:
            pred = pred[:, -1]
            true_ys = true_ys[:, -1]
        loss = criterion(pred, true_ys)

        total_loss += loss.item()

    return total_loss / test_batches
