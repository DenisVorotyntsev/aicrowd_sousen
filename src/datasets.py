from typing import Optional, Dict, List
import os
import collections
from multiprocessing import Pool
from pprint import pprint

import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import torch

from transform import spectral_augmentation, get_crop, get_crops, normalize, pad_to_desired_length
from augmentations import Compose, UseWithProb, OneOf, NoAugmentations, SpectralAugmentation, Mixup, CutMix, CutAway
from utils import get_data_files_from_folder


def collate_fn(list_arrays):
    """
    Y_ohe and y_ohe_fake are equal only for validation part; for train y_ohe - correct target, y_ohe_fake = not
    :param list_arrays:
    :return:
    """
    x = torch.cat([a["x"] for a in list_arrays], dim=0)
    # sample_weight = torch.cat([a["sample_weight"] for a in list_arrays], dim=0)
    y = torch.cat([a["y"] for a in list_arrays], dim=0)
    return x, y


def read_and_process_features(path) -> torch.Tensor:
    audio, sr = librosa.load(path, sr=None, mono=True)

    num_freq_bin = 128
    num_fft = 2048
    hop_length = num_fft // 2

    mels = librosa.feature.melspectrogram(audio, sr=sr, n_fft=num_fft,
                                          hop_length=hop_length, n_mels=num_freq_bin,
                                          fmin=0.0, fmax=sr/2, htk=True, norm=None)
    ts = librosa.power_to_db(mels, ref=0, top_db=120)
    ts = np.swapaxes(ts, 0, 1)
    return ts


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path_to_data: str,
                 files_to_use: Optional[List[str]] = None,
                 load_data_to_mem: bool = True,
                 path_to_targets: Optional[str] = None,
                 features_stats: Optional[Dict[str, any]] = None,
                 crop_params=None,
                 mode: str = "train"
                 ):

        self.path_to_data = path_to_data
        self.files_to_use = files_to_use
        self.load_data_to_mem = load_data_to_mem
        self.path_to_targets = path_to_targets
        self.features_stats = features_stats
        self.mode = mode  # "train", "val", "test"

        self.train_augmentation = UseWithProb(
            SpectralAugmentation(num_mask=2, freq_masking_range=(0.05, 0.15), time_masking_range=(0.05, 0.20)),
            prob=0.2
        )

        if crop_params is None:
            crop_params = {"crop_size": 120, "step": 60}
        self.crop_params = crop_params

        self.dataset_features_stats = collections.defaultdict(list)
        self.file_name_to_features_mapping = {}

        self._init()

    def _init(self):
        # prepare data files
        files_in_folder = get_data_files_from_folder(self.path_to_data)
        if self.files_to_use is not None:
            files = [f for f in self.files_to_use if f in files_in_folder]
        else:
            files = list(files_in_folder)
        self.files = files
        self.num_samples = len(self.files)

        # read targets
        if self.path_to_targets is not None:
            targets_df = pd.read_csv(self.path_to_targets)
            file_names = targets_df["wav_id"].astype("str") + ".wav"
            file_target = targets_df["label"]
            targets = dict(zip(file_names, file_target))
            self.targets = targets

        # read data to mem
        if self.load_data_to_mem:
            self.prepare_data_multi(self.files)
            self.prepare_dataset_features_stats()

        # calculate features stats to normalize data later
        if self.features_stats is None:
            self.features_stats = self.dataset_features_stats
            print("Using calculated features stats")
            print(self.features_stats)
        else:
            print("Using predefined features stats")
            print(self.features_stats)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_name = file_name.split(".")[0]  # remove file type

        # features
        x = self.get_features(idx)
        x = pad_to_desired_length(x, self.crop_params["crop_size"], mode="2d")

        if self.mode == "train":
            y = self.get_y(idx)
            x = get_crop(x, self.crop_params["crop_size"], seed=None)
            x, y = self.train_augmentation(x, y)
            x = normalize(x, self.features_stats["mean"], self.features_stats["std"])
            x = torch.Tensor(x).unsqueeze(0)
        elif self.mode == "val":
            y = self.get_y(idx)
            x = normalize(x, self.features_stats["mean"], self.features_stats["std"])
            x = get_crop(x, self.crop_params["crop_size"], self.crop_params["step"])
            x = torch.Tensor(x).unsqueeze(0)
        elif self.mode == "test":
            x = normalize(x, self.features_stats["mean"], self.features_stats["std"])
            x = get_crop(x, self.crop_params["crop_size"], self.crop_params["step"])
            x = torch.Tensor(x).unsqueeze(0)
            y = None
        else:
            raise ValueError(f"Not known mode {self.mode}")

        # sum of weights of samples from a single file must be equal to 1
        n_samples = x.size(0)
        sample_weight = torch.Tensor([1 / n_samples] * n_samples)
        file_name_list = [file_name] * n_samples

        results = {
            "x": x,
            "y": y,
            "sample_weight": sample_weight,
            "file_name": file_name_list,
        }
        return results

    def prepare_dataset_features_stats(self):
        print("Calculating features stats...")
        for file_name in tqdm(self.files):
            X = self.file_name_to_features_mapping.get(file_name)
            x_mean = X.mean()
            x_std = X.std()
            self.dataset_features_stats["mean"].append(x_mean)
            self.dataset_features_stats["std"].append(x_std)

        self.dataset_features_stats["mean"] = np.mean(self.dataset_features_stats["mean"])
        self.dataset_features_stats["std"] = np.mean(self.dataset_features_stats["std"])

        print("Dataset feature stats")
        pprint(self.dataset_features_stats)

    def get_features(self, idx):
        file_name = self.files[idx]
        X = self.file_name_to_features_mapping.get(file_name, None)
        if X is None:
            path = os.path.join(self.path_to_data, file_name)
            X = read_and_process_features(path)
        return X

    def get_y(self, idx):
        file_name = self.files[idx]
        y = self.targets[file_name]
        y = torch.Tensor([y]).long()
        return y

    def prepare_data_multi(self, files):
        print("Loading data in RAM...")
        paths = [os.path.join(self.path_to_data, f) for f in files]
        n_workers = os.cpu_count()

        with Pool(n_workers) as p:
            results = list(
                tqdm(
                    p.imap(read_and_process_features, paths), total=len(paths)
                )
            )
        self.file_name_to_features_mapping = dict(zip(files, results))
