"""geofm.datasets.transforms

Data augmentation transforms for geospatial imagery.
Applies identical transforms to both image and label tensors.
"""
import torch


class RandomHorizontalFlip:
    """Randomly flip image and label horizontally."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if torch.rand(1) < self.p:
            image = torch.flip(image, dims=[-1])
            label = torch.flip(label, dims=[-1])
        return image, label


class RandomVerticalFlip:
    """Randomly flip image and label vertically."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if torch.rand(1) < self.p:
            image = torch.flip(image, dims=[-2])
            label = torch.flip(label, dims=[-1])
        return image, label


class RandomRotation90:
    """Randomly rotate image and label by 0, 90, 180, or 270 degrees."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if torch.rand(1) < self.p:
            k = torch.randint(1, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[-2, -1])
            label = torch.rot90(label, k, dims=[-2, -1])
        return image, label


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


def build_train_transform(flip_p=0.5, rotation_p=0.5):
    """Build training transforms with augmentation.

    Args:
        flip_p: Probability for random flips (default: 0.5)
        rotation_p: Probability for random rotation (default: 0.5)

    Returns:
        Compose transform that operates on (image, label) tuples
    """
    return Compose([
        RandomHorizontalFlip(p=flip_p),
        RandomVerticalFlip(p=flip_p),
        RandomRotation90(p=rotation_p),
    ])


def build_val_transform():
    """Build validation transforms (no augmentation)."""
    return Compose([])