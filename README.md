# Fisher information and the geometry of memorization in neural networks

Experiment code for *Fisher information and the geometry of memorization in neural
networks* ([arXiv:2610.16161](https://arxiv.org/abs/2610.16161)) by Daniel T. Bernstein,
Chase Goddard, Luca Di Carlo, Vudtiwat Ngampruetikorn, and David J. Schwab.

We train models until they memorize data they cannot generalize from: Gaussian-noise
images or mislabeled CIFAR-10 images in a ResNet-20, and a small set of in-context
regression tasks in a transformer. We then compute the Fisher information per layer or
per component, and project the weights onto its top-`k` eigenvectors to see what each
subspace stores.

These scripts train the models and generate the data behind the paper's figures and tables,
and each guide lists the files for each figure. The repository holds code only, and it stops
at the data, with no plotting code. The barcode model of Figure 6 has no code here: its
derivation in the paper is the artifact.

## Layout

| Path | Contents |
| --- | --- |
| [`src/`](src) | Models, data, and Fisher computations |
| [`scripts/`](scripts) | One entry point per experiment |
| [`docs/`](docs) | A guide for each experiment, and numerical notes |

## Setup

Use Python 3.12. The experiments ran on CUDA 12.1, and everything beyond the smallest
layers needs a GPU.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -c "from torchvision.datasets import CIFAR10; CIFAR10('data', download=True)"
```

The last line fetches CIFAR-10 into `data/` up front, since compute nodes often have no
network access. Run every command from the repository root.

## Reproducing the paper

| Result | Scripts, in order | Guide |
| --- | --- | --- |
| Figure 1: input-noise layer projections | `train_gaussian.py` → `project_gaussian_layerwise.py` | [Input noise](docs/input-noise.md) |
| Figure 2: label-noise layer projections | `train_label_noise.py` → `project_label_noise_layerwise.py` | [Label noise](docs/label-noise.md) |
| Figure 3: block-diagonal Fisher projections | `train_label_noise.py` → `project_label_noise_global.py` | [Label noise](docs/label-noise.md) |
| Figure 4: in-context learning projections | `train_icl.py` → `project_icl_components.py` → `aggregate_icl_components.py` → `export_icl_main_panels.py` | [In-context learning](docs/icl.md) |
| Figure 5: single-sample contributions and projections | `project_gaussian_layerwise.py` → `pr_gaussian.py`; `pr_label_noise.py`; `project_single_sample_{gaussian,label_noise}.py` → `summarize_single_sample.py` | [Input noise](docs/input-noise.md), [Label noise](docs/label-noise.md) |
| Figure 7: all input-noise layers | `train_gaussian.py` → `project_gaussian_layerwise.py` | [Input noise](docs/input-noise.md) |
| Figure 8: all transformer blocks | `train_icl.py` → `project_icl_components.py` → `aggregate_icl_components.py` → `export_icl_appendix_panels.py` | [In-context learning](docs/icl.md) |
| Figure 9: all layers, clean Fisher | `train_label_noise.py` → `project_label_noise_layerwise.py` | [Label noise](docs/label-noise.md) |
| Figure 10: all layers, full Fisher | `train_label_noise.py` → `project_label_noise_layerwise.py` | [Label noise](docs/label-noise.md) |
| Figure 11: SVD and random controls | `project_gaussian_{layerwise,svd,random}.py`; `subspace_controls.py` | [Input noise](docs/input-noise.md#controls-figure-11), [Label noise](docs/label-noise.md#controls-figure-11) |
| Tables 1, 2: accuracy after sample projection | `project_single_sample_{gaussian,label_noise}.py` → `summarize_single_sample.py` | [Input noise](docs/input-noise.md#single-sample-projections-figure-5-tables-1-and-2) |

The [numerical notes](docs/numerics.md) cover the details that matter for matching the
numbers, including an eigensolver limit that fails silently.

## Running the experiments

The guides give each experiment as plain commands that run one after another. A full run is
long, mostly because of the input-noise sweeps, so on a cluster submit each pass through a
loop as its own job, except where a guide says otherwise. Each guide lists the memory and
time every step took on A100 GPUs. Everything fits on one 80 GB GPU with 120 GB of host
memory, and most steps fit in a 10 GB slice of a GPU.

## Citation

```bibtex
@article{bernstein2026fisher,
  title   = {Fisher information and the geometry of memorization in neural networks},
  author  = {Bernstein, Daniel T. and Goddard, Chase and Di Carlo, Luca and
             Ngampruetikorn, Vudtiwat and Schwab, David J.},
  journal = {arXiv preprint arXiv:2610.16161},
  year    = {2026}
}
```

## Attribution

The ResNet follows [He et al. (2015)](https://arxiv.org/abs/1512.03385), adapted from
Yerlan Idelbayev's implementation. The in-context learning code comes from
[Goddard et al. (ICML 2025)](https://arxiv.org/abs/2506.05574), and its attention follows
Andrej Karpathy's lecture and nanoGPT. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for the upstream licenses. The input-noise and single-sample experiments were contributed by
Daniel Bernstein.

## License

MIT. See [LICENSE](LICENSE).
