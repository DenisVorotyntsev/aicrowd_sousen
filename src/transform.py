from typing import Optional, Dict, List, Tuple
import copy

import random
import numpy as np


def pad_to_desired_length(ts, desired_length: int = 1_000, mode="1d"):
    ts_length = ts.shape[0]
    if ts_length < desired_length:
        delta = desired_length - ts_length

        if mode == "1d":
            to_pad = (0, delta)
        elif mode == "2d":
            to_pad = ((0, delta), (0, 0))
        else:
            raise ValueError(f"Now known model {mode}")

        ts = np.pad(ts, to_pad, mode="wrap")
    return ts


def get_crop(x, crop_size: int = 512, seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
    i_start = random.randint(0, x.shape[0] - crop_size)
    i_end = i_start + crop_size
    x_crop = x[i_start:i_end, ...]
    return x_crop


def get_crops(x, crop_size: int = 512, step: int = None):
    """
    Create crops from a given input.
    :param x: input mel
    :param crop_size: size of crop
    :param step: frequency of crops,
        step > crop_size -> hole between crops
        step < crop_size -> overlap between crops
    :return:
    """
    if step is None:
        step = int(crop_size * 0.9)

    if x.shape[0] < (crop_size + step):
        i_starts = [0, x.shape[0] - crop_size]
    else:
        i_starts = list(range(0, x.shape[0], step))

    x_crops = []
    for i_start in i_starts:
        i_end = i_start + crop_size
        x_crop = x[i_start:i_end, ...]
        if x_crop.shape[0] == crop_size:
            x_crop = x_crop[np.newaxis, ...]
            x_crops.append(x_crop)
    x_crops = np.concatenate(x_crops, axis=0)
    return x_crops


def normalize(x, x_mean=None, x_std=None, eps: float = 1e-6):
    if x_mean is None:
        x_mean = x.mean()
    if x_std is None:
        x_std = x.std()
    x_normed = (x - x_mean + eps) / (x_std + eps)
    return x_normed


def spectral_augmentation(
        spec: np.ndarray,
        num_mask: int = 2,
        freq_masking_range=(0.05, 0.15),
        time_masking_range=(0.05, 0.20),
        value=0
):
    # spec shape: time, channels
    spec = copy.deepcopy(spec)
    all_frames_num, all_freqs_num = spec.shape

    for i in range(num_mask):
        if freq_masking_range is not None:
            freq_percentage = random.uniform(*freq_masking_range) / num_mask
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = value

        if time_masking_range is not None:
            time_percentage = random.uniform(*time_masking_range) / num_mask
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = value
    return spec

