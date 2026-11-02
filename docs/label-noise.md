# Label noise

ResNet-20 on CIFAR-10 with 10% of the training labels corrupted. Run every command from
the repository root.

## Training

```bash
python scripts/train_label_noise.py --noise-fraction 0.10 --noise-seed 42 --seed 42 --save-dir output/noisy10pct
```

This trains for 200 epochs with SGD at learning rate 0.1, reduced 10x at epochs 100 and
150, with batch size 128, momentum 0.9, and weight decay 1e-4, and writes `model.pt` and
`noise_metadata.npz`. Noise seed 42 selects the corrupted labels of the paper's model. Its
initialization seed was not recorded, so retraining gives an equivalent model, not the same
weights.

## Per-layer projections (Figures 2, 9, 10)

For each convolutional or linear layer, this builds the Fisher-scaled Jacobian,
eigendecomposes `G = J Jᵀ` in float64, and sweeps `k`, projecting the weights onto the
top-`k` eigenvectors, re-estimating BatchNorm, and evaluating four data splits.

```bash
python scripts/project_label_noise_layerwise.py --model-dir output/noisy10pct --fisher-set clean --prefix proj
python scripts/project_label_noise_layerwise.py --model-dir output/noisy10pct --fisher-set full --prefix proj_full
```

To run one layer at a time, add for example `--layers layer3.2.conv1`. A sweep skips any layer
whose output file already exists, so each seed needs its own prefix.

Figure 2 also needs these runs, where `--seed` redraws the Fisher sample on the same model:

```bash
for seed in 43 44; do
  python scripts/project_label_noise_layerwise.py --model-dir output/noisy10pct \
    --layers conv1,layer3.2.conv1 --fisher-set clean --prefix "proj_s$seed" --seed "$seed"
  python scripts/project_label_noise_layerwise.py --model-dir output/noisy10pct \
    --layers conv1,layer3.2.conv1 --fisher-set full --prefix "proj_full_s$seed" --seed "$seed"
done
```

## Block-diagonal projections (Figure 3)

Treating the Fisher as block diagonal over layers makes its eigenpairs the union of the
per-layer eigenpairs. A global top-`k` subspace therefore comes from pooling all 268,346
eigenvalues and giving each layer however many of the first `k` are its own.

```bash
python scripts/project_label_noise_global.py --model-dir output/noisy10pct --fisher-set clean --prefix global_clean --reverse
python scripts/project_label_noise_global.py --model-dir output/noisy10pct --fisher-set full --prefix global_full --reverse
```

`--reverse` eigendecomposes the largest blocks first, so a memory failure comes early.

## Participation ratios (Figure 5)

```bash
python scripts/pr_label_noise.py --model-dir output/noisy10pct --n 0
```

The first run also computes the first layer's Fisher and caches it as `fisher_matrix_0.pt`.

## Controls (Figure 11)

This repeats the sweep on weight-SVD and Haar-random subspaces, alongside the Fisher:

```bash
for layer in conv1 layer3.2.conv1; do
  for seed in 1 2 3; do
    for method in fisher svd random; do
      python scripts/subspace_controls.py --model-dir output/noisy10pct \
        --layer "$layer" --method "$method" --seed "$seed"
    done
  done
done
```

## Single-sample projections (Figure 5, Tables 1 and 2)

See the [input-noise guide](input-noise.md#single-sample-projections-figure-5-tables-1-and-2),
which covers both models.

## Outputs

Each sweep writes its NPZ to the model directory, with `k_values` and, for each split, a loss
array `<split>` and an accuracy array `<split>_acc`:

| Split | Samples and labels |
| --- | --- |
| `clean_train` | Uncorrupted training samples, true labels |
| `noised_noisedlbl` | Corrupted samples, their assigned (wrong) labels |
| `noised_truelbl` | Corrupted samples, their true labels |
| `test` | CIFAR-10 test set |

`PR_noisy_layerwise_0.pt` holds `psi_all`, `psi_uncorrupted`, and `psi_corrupted`, ordered from
the top eigenvalue down. Load it with `torch.load(path, weights_only=False)`.

These are the files for each figure, in `output/noisy10pct/`, where `<layer>` is written with
underscores, as in `layer3_2_conv1`:

| Figure | Files |
| --- | --- |
| Figure 2: label-noise layer projections | `proj_<layer>.npz`, `proj_full_<layer>.npz`, `proj_s<seed>_<layer>.npz`, and `proj_full_s<seed>_<layer>.npz` |
| Figure 3: block-diagonal Fisher projections | `global_clean_proj.npz`, `global_full_proj.npz` |
| Figure 5: single-sample contributions and projections | `PR_noisy_layerwise_0.pt` |
| Figure 9: all layers, clean Fisher | `proj_<layer>.npz` |
| Figure 10: all layers, full Fisher | `proj_full_<layer>.npz` |
| Figure 11: SVD and random controls | `controls/fisher_clean_<layer>_seed<N>.npz`, `controls/svd_<layer>_seed<N>.npz`, and `controls/random_<layer>_seed<N>.npz` |

## Resources

Peak GPU memory is set by the Jacobian and eigendecomposition of the largest layer in a run.

| Step | GPU memory | Host memory | Time on an A100 |
| --- | --- | --- | --- |
| Training | Under 2 GB | 24 GB | 8 minutes |
| One layer, `conv1`, `layer1.*`, `layer2.*`, or `linear` (`P` ≤ 9,216) | 4.1 GiB | 16 GB | 1–3 minutes |
| One layer, `layer3.0.conv1` (`P` = 18,432) | 13.6 GiB | 32 GB | 1 minute |
| One layer, `layer3.0.conv2`, `layer3.1.*`, or `layer3.2.*` (`P` = 36,864) | 54+ GiB | 48 GB | 3 minutes |
| Block-diagonal sweep | 80 GB GPU | 120 GB | 14 minutes |
| One control run | 80 GB GPU at `layer3.2.conv1` | 64 GB | 1–4 minutes |
| Participation ratios, first layer | 10 GB | 32 GB | 2.5 hours |

All twenty layers with both Fisher sets take under two GPU hours. The block-diagonal sweep
holds all twenty eigenbases in host memory, about 61 GB.
