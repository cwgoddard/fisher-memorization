# Numerical notes

## The eigensolver limit

`torch.linalg.eigh` on CUDA returns at most 32,768 = 2^15 eigenvectors with the versions
pinned here. Past that, the extra columns come back identically zero, with no error and
with plausible eigenvalues alongside them. On ResNet-20's five `P = 36,864` layers that
loses 4,096 columns each, and `U Uᵀ` then projects onto the recovered block rather than the
identity, so a sweep never returns to the trained weights at `k = P`. Other library
versions may fail differently, so check yours.

The two workstreams handle this differently:

- Label noise: `fisher_utils.complete_basis` rebuilds the missing columns. They lie in the
  numerically null block, where any orthonormal completion is an equally valid eigenbasis.
- Input noise: `fisher_projection._eigh_symmetric` switches to SciPy's MRRR driver on CPU
  above the limit.

`scripts/check_eigh_size.py` reproduces the boundary on your GPU and library versions, and
needs an 80 GB GPU. A complete basis does not by itself establish that eigenpairs are
correct, so check two diagnostics: `recon_err`, the residual of the `k = P` projection, which
the label-noise sweeps report and warn on above 1e-8, and `lambda_check` in the
participation-ratio outputs.

## Independent Fisher samples

`train_label_noise.py` picks the corrupted labels with
`RandomState(seed).choice(50000, k, replace=False)`, and legacy `RandomState.choice` with
`replace=False` is `permutation(pop)[:k]`. Drawing the Fisher sample the same way, from the
same population with the same seed, returns a prefix of the corrupted index list. Whenever
`n <= k`, a nominal "full training set" Fisher would then be 100% mislabeled.
`fisher_utils.draw_fisher_samples` therefore uses a separate PCG64 stream, and warns if the
sampled corrupted fraction departs from the pool's by more than 6σ.

## Fisher definitions

- The vision Fisher is an expectation over the model's own predictive distribution, so it
  never reads the target labels. See [the ICL guide](icl.md#behavior-to-be-aware-of) for
  how the transformer's estimate differs.
- Label-noise Jacobians differentiate `2*sqrt(softmax(logits))` in float32. All later linear
  algebra runs in float64.
- Label-noise Fisher subsets have `P // 10 + 1` samples per layer. Each ten-class sample
  contributes rank at most nine, so this does not guarantee a full-rank Fisher.
- Block-diagonal projections cover convolutional and linear parameters, and leave the
  BatchNorm affine parameters fixed. Single-sample projections target convolutional and
  linear weights only, so not the linear layer's bias.

## Participation ratios

`pr_gaussian.py` and `pr_label_noise.py` compute the paper's restricted ratio, which keeps
an all-sample denominator: `Ψ_G^k = (Σ_{i∈G} c_i^k)² / (n Σ_i (c_i^k)²)`. Because of cross
terms, `Ψ^k` is generally not the sum of the restricted ratios. Each script also checks the
identity `Σ_i c_i^k = n λ_k` and saves the result as `lambda_check`.

## BatchNorm

Statistics are stale after any change to the weights, so every evaluation follows a
recalibration pass.

## Reproducibility

Results reproduce in distribution, not bitwise. Rerunning the label-noise sweeps with the
same seeds changed accuracies by about 1e-3, which we attribute to float32 Jacobians being
nondeterministic on GPU. The SVD control, which uses no Jacobian, came back bitwise
identical. `torch.randn` with a fixed seed also gives different streams across GPU
configurations, so a random-basis seed is not portable: compare a rerun against the spread
across seeds, not against one curve.
