import random

import numpy as np

from transform import pad_to_desired_length, get_crop, spectral_augmentation


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for transform in self.transforms:
            x, y = transform(x, y)
        return x, y


class UseWithProb:
    def __init__(self, transform, prob=0.42):
        self.transform = transform
        self.prob = prob

    def __call__(self, x, y):
        if random.random() < self.prob:
            x, y = self.transform(x, y)
        return x, y


class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, x, y):
        transform = np.random.choice(self.transforms, p=self.p)
        x, y = transform(x, y)
        return x, y


class NoAugmentations:
    def __call__(self, x, y):
        return x, y


class SpectralAugmentation:
    def __init__(self, num_mask=2, freq_masking_range=(0.05, 0.15), time_masking_range=(0.05, 0.20), value=None):
        self.num_mask = num_mask
        self.freq_masking_range = freq_masking_range
        self.time_masking_range = time_masking_range
        self.value = value

    def __call__(self, x, y):
        x = spectral_augmentation(x, self.num_mask, self.freq_masking_range, self.time_masking_range, x.min())
        return x, y


class Mixup:
    def __init__(self, x2, y2, mixup_contribution_range=(0.1, 0.5)):
        self.x2 = x2
        self.y2 = y2
        self.mixup_contribution = np.random.uniform(*mixup_contribution_range)

    def __call__(self, x, y):
        x = x * (1 - self.mixup_contribution) + self.x2 * self.mixup_contribution
        y = y * (1 - self.mixup_contribution) + self.y2 * self.mixup_contribution
        return x, y


class CutMix:
    def __init__(self, x2, y2, cutmix_contribution_range=(0.1, 0.5)):
        self.x2 = x2
        self.y2 = y2
        self.cutmix_contribution = np.random.uniform(*cutmix_contribution_range)

    def __call__(self, x, y):
        cutmix_len = int(len(x) * self.cutmix_contribution)
        i_start = random.randint(0, x.shape[0] - cutmix_len)
        i_end = i_start + cutmix_len
        x[i_start: i_end, :] = get_crop(self.x2, crop_size=cutmix_len)
        y = y * (1 - self.cutmix_contribution) + self.y2 * self.cutmix_contribution
        return x, y


class CutAway:
    def __init__(self, num_cuts: int = 2, cutaway_fraction=(0.1, 0.25)):
        self.num_cuts = num_cuts
        self.cutaway_fraction = cutaway_fraction

    def __call__(self, x, y):
        initial_x_len = len(x)
        for _ in range(self.num_cuts):
            cut_fraction = np.random.uniform(*self.cutaway_fraction) / self.num_cuts
            cut_len = int(initial_x_len * cut_fraction)
            i_start = random.randint(0, x.shape[0] - cut_len)
            i_end = i_start + cut_len
            x = np.vstack([
                x[:i_start, :], x[i_end:, :]
            ])
        x = pad_to_desired_length(x, desired_length=initial_x_len, mode="2d")
        return x, y

