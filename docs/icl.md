# In-context learning

A three-block GPT-2 style transformer is trained on contexts `(x_1, y_1, ..., x_n)` with
`y_i = w_t · x_i`, where each context's task vector `w_t` comes from a fixed set of 16.

The model has hidden dimension 32, 8 heads, and dropout 0.1. Contexts hold 20 pairs with
`x` in `R^5`. Training runs 50,000 steps at learning rate 5e-4 with weight decay 1e-4 and
batch size 128.

## Commands (Figures 4, 8)

Run from the repository root:

```bash
for seed in {0..16}; do
  python scripts/train_icl.py --seed "$seed" --output-dir results/icl \
    --d-model 32 --total-steps 50000 --skip-fisher
  python scripts/project_icl_components.py --results-dir results/icl --seed "$seed" \
    --ensemble-alpha 1.0 --output-name fisher_component_projection_wse_a1p0
done
python scripts/aggregate_icl_components.py --results-dir results/icl
python scripts/export_icl_main_panels.py --results-dir results/icl --alpha 1.0
python scripts/export_icl_appendix_panels.py --results-dir results/icl --alpha 1.0
```

Pass `--d-model` and `--total-steps` explicitly, since their defaults are 64 and 120,000.
`--skip-fisher` skips an additional whole-block sweep that no result uses.

## Outputs

The last two commands write one NPZ per panel to `results/icl/`:

| Figure | Files |
| --- | --- |
| Figure 4: in-context learning projections | `block1_panel_npz/panel_*.npz` |
| Figure 8: all transformer blocks | `appendix_panel_npz/panel_*.npz` |

Each holds `ks`, `train_mean`, `train_std`, `test_mean`, and `test_std`, with the reference
levels `ref_train_mean`, `ref_test_mean`, and `early_stopping_test_mean`. The exception is
`panel_A_training_curve.npz`, which holds `log_steps`, `train_task_loss_history`, and
`held_out_loss_history`, with the best and final steps.

## Reading the code

- **Alpha.** `project_icl_components.py` replaces a component with
  `(1 - alpha) * w_orig + alpha * w_proj(k)`. The commands use `alpha = 1`, pure projection.
  Exporters select the nearest available alpha, so check the value they save.
- **Block numbers.** Blocks are zero-based in the code and one-based in the paper, so `block1`
  is the paper's Block 2.

## Behavior to be aware of

These are kept as they were, because changing them would change the published numbers.

- **Attention dropout stays on during evaluation.** `src/transformer.py` passes a fixed
  `dropout_p` to `scaled_dot_product_attention`, which applies it regardless of
  `model.eval()`. Dropout elsewhere in the model does respect `eval()`.
- **The curvature estimate is empirical.** A component's Fisher averages outer products of
  batch-mean MSE gradients with dropout active, using one gradient per parameter in the
  component. It is not an expectation over a predictive distribution. A sweep therefore
  costs as many backward passes as the component has parameters.

## Resources

Training and projection each fit in a 10 GB slice of a GPU with 32 GB of host memory, and each
takes under an hour per seed on an A100.
