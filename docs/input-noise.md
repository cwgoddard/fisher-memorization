# Input noise and single-sample projections

ResNet-20 on CIFAR-10 plus 500 Gaussian-noise images, moment-matched to CIFAR-10 classes
and given random labels. Run every command from the repository root.

## Training

```bash
for run in 004 005 006 007; do
  python scripts/train_gaussian.py --gaussian-count 500 \
    --run-name "resnet20_cifar10gaussian_g500_randlabels_run$run"
done
```

Training uses the same schedule as the [label-noise model](label-noise.md#training).
Always pass `--gaussian-count 500`, since the default is 10,000. The paper's models were
trained without a fixed seed, so retraining gives equivalent models, not the same weights.
`--seed` seeds PyTorch only, while the Gaussian images' classes and labels also draw from NumPy.

## Per-layer projections (Figures 1, 7)

```bash
for run in 004 005 006 007; do
  model="experiments/resnet20_cifar10gaussian_g500_randlabels_run$run"
  for depth in {0..12} 19; do
    python scripts/project_gaussian_layerwise.py --model-path "$model" --scope "$depth" --num-k-points 50
  done
  for depth in {13..18}; do
    python scripts/project_gaussian_layerwise.py --model-path "$model" --scope "$depth" --num-k-points 500
  done
done
```

Each run writes `bulk_fisher_experiment_<depth>_<n>.pt` and `fisher_matrix_<depth>.pt` to the
model directory, where `<n>` counts up from 1.

## Participation ratios (Figure 5)

This reads the `fisher_matrix_0.pt` written above:

```bash
python scripts/pr_gaussian.py --n 0 --model-path experiments/resnet20_cifar10gaussian_g500_randlabels_run004
```

## Single-sample projections (Figure 5, Tables 1 and 2)

These use `run004` and the [label-noise model](label-noise.md#training). For each of
`k = 0, ..., 9`, and each sample in the Fisher sets, they project the top `k` eigenvectors of
that sample's Fisher out of the model's weights, then measure the sample and the wider
population. `k = 0` keeps the weights unchanged, as a matched BatchNorm control.

```bash
for k in {0..9}; do
  python scripts/project_single_sample_gaussian.py \
    --model-path experiments/resnet20_cifar10gaussian_g500_randlabels_run004 \
    --output-dir sanity_checks/cumulative_bn_v1/single_sample/gaussian --output-tag cumulative_bn_v1 \
    --top-k "$k" --num-gaussian-fisher 500 --num-calib 5000 --batch-size 250 --calib-batches 999999
  python scripts/project_single_sample_label_noise.py \
    --model-dir output/noisy10pct \
    --output-dir sanity_checks/cumulative_bn_v1/single_sample/noisy --output-tag cumulative_bn_v1 \
    --top-k "$k" --num-calib 5000 --calib-source all --batch-size 250 --calib-batches 999999
done
python scripts/summarize_single_sample.py
```

The last command prints both tables and the self-hit rates for Figure 5. The calibration set
is 5,000 held-out CIFAR-10 images plus all 500 Gaussian images for the input-noise model, and
4,500 clean plus 500 corrupted images for the label-noise model.

## Controls (Figure 11)

```bash
model=experiments/resnet20_cifar10gaussian_g500_randlabels_run004
for seed in 1 2 3; do
  python scripts/project_gaussian_layerwise.py --model-path "$model" --scope 0 --num-k-points 50 \
    --bn-averaging cumulative --output-tag analysis --seed "$seed"
  python scripts/project_gaussian_svd.py --model-path "$model" --scope 0 --seed "$seed"
  python scripts/project_gaussian_random.py --model-path "$model" --scope 0 --num-k-points 50 --seed "$seed"
done
```

Run this loop in one job, since each script numbers its outputs by counting the files already
there, and parallel runs would collide. For SVD, `k` is the rank times the number of columns of
the weight matrix.

## Optional check

`check_bn_calibration.py` tests whether BatchNorm statistics and accuracy have stabilized after
1, 2, 5, 10, and all calibration batches, with unchanged weights. No figure or table reads its
output.

```bash
for experiment in gaussian noisy; do
  python scripts/check_bn_calibration.py --experiment "$experiment" \
    --output-dir sanity_checks/cumulative_bn_v1/calibration_convergence
done
```

## Outputs

Results are in the model directories under `experiments/`. Load `.pt` files with
`torch.load(path, weights_only=False)`. Projection runs hold `["on_projector"][split]["accuracy"]`,
a dictionary from `k` to accuracy, for the splits `train`, `test`, and `gauss`. Participation
ratios hold `psi_all`, `psi_cifar`, and `psi_gauss`, ordered from the top eigenvalue down.

| Figure | Files |
| --- | --- |
| Figure 1: input-noise layer projections | `bulk_fisher_experiment_0_1.pt` and `bulk_fisher_experiment_19_1.pt` from all four models |
| Figure 5: single-sample contributions and projections | `PR_gauss_layerwise_0.pt` from `run004`, and the self-hit rates from `summarize_single_sample.py` |
| Figure 7: all input-noise layers | `bulk_fisher_experiment_<depth>_1.pt` for every depth, from all four models |
| Figure 11: SVD and random controls | `bulk_fisher_experiment_analysis_0_<n>.pt`, `bulk_svd_projection_experiment_0_<n>.pt`, and `bulk_random_projection_experiment_0_<n>.pt` from `run004` |
| Tables 1, 2: accuracy after sample projection | Printed by `summarize_single_sample.py` |

## Resources

| Step | GPU memory | Host memory | Time on an A100 |
| --- | --- | --- | --- |
| Training | 10 GB | 32 GB | 1.7 hours per model |
| Projections, depths 0–12 | 10 GB | 32 GB | 2–3 hours per depth |
| Projections, depth 19 | 10 GB | 32 GB | 25 minutes |
| Projections, depth 13 | Full GPU | 60 GB | 4 hours |
| Projections, depths 14–18 | Full GPU | 60 GB | 9.5 hours per depth |
| Participation ratios, first layer | 10 GB | 32 GB | 20 minutes |
| Single-sample, input noise | 10 GB | 32 GB | 1.7 hours per `k` |
| Single-sample, label noise | 10 GB | 32 GB | 2.8 hours per `k` |
| Controls | 10 GB | 32 GB | 3 hours per seed |

One model's full per-layer sweep takes about 80 GPU hours. At `P` = 36,864 the Fisher exceeds
the GPU eigensolver's limit and uses SciPy on CPU ([details](numerics.md#the-eigensolver-limit)).
