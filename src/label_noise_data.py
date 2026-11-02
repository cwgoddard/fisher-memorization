"""CIFAR-10 with synthetic label noise."""

import copy
import os

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

NUM_CLASSES = 10

# Splits every projection sweep evaluates.
SPLITS = ["clean_train", "noised_noisedlbl", "noised_truelbl", "test"]


def load_cifar10(data_root="./data", download=True):
    """CIFAR-10 train and test sets, normalized, no augmentation."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_ds = datasets.CIFAR10(data_root, train=True, transform=transform, download=download)
    test_ds = datasets.CIFAR10(data_root, train=False, transform=transform, download=download)
    return train_ds, test_ds


def apply_label_noise(dataset, noise_fraction, seed=42, num_classes=NUM_CLASSES):
    """Reassign a fraction of labels uniformly among the wrong classes.

    Mutates dataset.targets and returns the metadata that every downstream
    script reads back from noise_metadata.npz.
    """
    rng = np.random.RandomState(seed)
    true_labels = np.array(dataset.targets, dtype=np.int64)
    N = len(true_labels)

    noised_idx = rng.choice(N, size=int(N * noise_fraction), replace=False)
    is_noised = np.zeros(N, dtype=bool)
    is_noised[noised_idx] = True

    noised_labels = true_labels.copy()
    for i in noised_idx:
        new_label = rng.randint(0, num_classes - 1)
        if new_label >= true_labels[i]:
            new_label += 1
        noised_labels[i] = new_label

    dataset.targets = noised_labels.tolist()
    return {
        "true_labels": true_labels,
        "noised_labels": noised_labels,
        "is_noised": is_noised,
        "noise_fraction": noise_fraction,
        "seed": seed,
    }


def load_experiment(model_dir, data_root="./data"):
    """Datasets, split indices and noise metadata for a trained model.

    The four evaluation splits are:
        clean_train       uncorrupted training samples, true labels
        noised_noisedlbl  corrupted samples against their training labels
        noised_truelbl    the same samples against their true labels
        test              CIFAR-10 test set
    """
    train_ds, test_ds = load_cifar10(data_root)
    with np.load(os.path.join(model_dir, "noise_metadata.npz")) as meta:
        is_noised = meta["is_noised"]
        noised_labels = meta["noised_labels"]
        noise_fraction = float(meta["noise_fraction"])

    train_noised = copy.copy(train_ds)
    train_noised.targets = noised_labels.tolist()

    clean_idx = np.where(~is_noised)[0]
    noised_idx = np.where(is_noised)[0]

    return dict(
        train=train_ds,
        train_noised=train_noised,
        test=test_ds,
        is_noised=is_noised,
        clean_idx=clean_idx,
        noised_idx=noised_idx,
        noise_fraction=noise_fraction,
        splits={
            "clean_train": (train_ds, clean_idx),
            "noised_noisedlbl": (train_noised, noised_idx),
            "noised_truelbl": (train_ds, noised_idx),
            "test": (test_ds, np.arange(len(test_ds))),
        },
    )
