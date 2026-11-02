"""Data loading and synthetic data generation for CIFAR-10 experiments."""

import numpy as np
import torch
import torch.utils.data as tdata
import torchvision.datasets as datasets
import torchvision.transforms as transforms

CIFAR10_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

CIFAR10_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        CIFAR10_NORMALIZE,
    ]
)

NUM_CLASSES = 10
IMAGE_SHAPE = (3, 32, 32)


class TensorDatasetWithIntLabels(tdata.Dataset):
    """Dataset wrapping tensors, returning labels as Python ints.

    This matches the label type returned by torchvision CIFAR-10, which
    is necessary for compatibility with ConcatDataset and standard
    training loops.
    """

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx].item())


def load_cifar10(data_root="./data"):
    """Load CIFAR-10 train and test sets with ImageNet-style normalization.

    ToTensor → Normalize(ImageNet mean/std), no data augmentation.
    """
    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, transform=CIFAR10_TRANSFORM, download=False
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, transform=CIFAR10_TRANSFORM, download=False
    )
    return train_dataset, test_dataset


def compute_class_statistics(dataset, num_classes=NUM_CLASSES):
    """Compute per-class, per-pixel mean and std from a dataset.

    Loads the entire dataset into memory to compute statistics. This
    operates on the already-transformed (normalized) images.
    """
    loader = tdata.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    all_images, all_labels = next(iter(loader))

    class_means = torch.empty((num_classes, *IMAGE_SHAPE))
    class_stds = torch.empty_like(class_means)

    for class_idx in range(num_classes):
        mask = all_labels == class_idx
        class_images = all_images[mask]
        class_means[class_idx] = class_images.mean(dim=0)
        class_stds[class_idx] = class_images.std(dim=0)

    return class_means, class_stds


def generate_moment_matched_gaussian(
    class_means,
    class_stds,
    num_samples,
    random_labels=True,
):
    """Generate Gaussian noise samples moment-matched to CIFAR-10 classes.

    For each sample, a source class is drawn uniformly at random; the
    pixel values are sampled from N(mean[source_class], std[source_class])
    independently per pixel.

    When random_labels=True, the assigned label is drawn independently
    of the source class (so pixel statistics and label are decoupled).
    When random_labels=False, the label equals the source class.
    """
    images = torch.empty((num_samples, *IMAGE_SHAPE))
    labels = torch.empty(num_samples, dtype=torch.long)
    source_classes = torch.empty(num_samples, dtype=torch.long)

    for i in range(num_samples):
        src_class = np.random.randint(0, NUM_CLASSES)
        images[i] = torch.normal(class_means[src_class], class_stds[src_class])
        source_classes[i] = src_class

        if random_labels:
            labels[i] = np.random.randint(0, NUM_CLASSES)
        else:
            labels[i] = src_class

    return {
        "images": images,
        "labels": labels,
        "source_classes": source_classes,
    }


def build_combined_dataset(data_root="./data", gaussian_count=500, random_labels=True):
    """Build a CIFAR-10 + Gaussian noise combined training set.

    Steps:
    1. Load CIFAR-10 train/test
    2. Compute per-class statistics from the full training set
    3. Generate a full-sized (50k) Gaussian dataset
    4. Randomly subsample `gaussian_count` samples
    5. Concatenate the subsample with CIFAR-10 train
    """
    cifar_train, cifar_test = load_cifar10(data_root)

    class_means, class_stds = compute_class_statistics(cifar_train)

    full_gaussian = generate_moment_matched_gaussian(
        class_means,
        class_stds,
        num_samples=len(cifar_train),
        random_labels=random_labels,
    )
    full_gaussian_dataset = TensorDatasetWithIntLabels(
        full_gaussian["images"], full_gaussian["labels"]
    )

    n_gaussian = min(gaussian_count, len(full_gaussian_dataset))
    subset_indices = torch.randperm(len(full_gaussian_dataset))[:n_gaussian]
    gaussian_subset = tdata.Subset(full_gaussian_dataset, subset_indices)

    subset_source_classes = full_gaussian["source_classes"][subset_indices]
    subset_labels = full_gaussian["labels"][subset_indices]
    subset_images = full_gaussian["images"][subset_indices]

    gaussian_metadata = {
        "indices": subset_indices.clone(),
        "images": subset_images,
        "labels": subset_labels,
        "source_classes": subset_source_classes,
        "is_moment_matched": subset_source_classes == subset_labels,
    }

    combined_train = tdata.ConcatDataset([cifar_train, gaussian_subset])

    return {
        "train_dataset": combined_train,
        "test_dataset": cifar_test,
        "cifar_train": cifar_train,
        "gaussian_subset": gaussian_subset,
        "gaussian_metadata": gaussian_metadata,
    }
