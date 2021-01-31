import os
import sys

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from datasets import CustomDataset
from utils import get_data_files_from_folder


def make_prediction(model, dataset, mode: str = "mean") -> np.ndarray:
    device = next(model.parameters()).device
    y_hat = []
    for batch in tqdm(dataset, total=len(dataset)):
        x = batch["x"]
        x = x.to(device)

        logits = model(x)

        if mode == "mean":
            logits = torch.mean(logits, dim=0)
        elif mode == "max":
            logits = torch.max(logits, dim=0).values
        elif mode == "sum":
            logits = torch.sum(logits, dim=0)
        else:
            pass

        probs = torch.nn.functional.softmax(logits)
        probs = probs.detach().cpu().numpy()

        y_hat.append(probs)
        del x
    y_hat = np.vstack(y_hat)
    return y_hat


def test(
    path_to_test_data: str = None,
    path_to_models: str = None,
    path_to_save_results: str = None,
):
    if path_to_test_data is None:
        path_to_test_data = "./data/external/val"
        path_to_save_results = "solution.csv"
        path_to_models = "./models/"

    # params of dataset and normalization
    num_classes = 3
    features_stats = {"mean": 86, "std": 22}

    # get test files and make predictions
    test_files = get_data_files_from_folder(path_to_test_data, data_type=".wav")
    test_files_names = [f.split(".")[0] for f in test_files]
    test_dataset = CustomDataset(
        path_to_data=path_to_test_data,
        files_to_use=test_files,
        load_data_to_mem=True,
        path_to_targets=None,
        features_stats=features_stats,
        mode="test"
    )

    # make predictions using several models
    models_paths = [os.path.join(path_to_models, f) for f in os.listdir(path_to_models) if f.endswith(".pt")]
    y_hat = np.zeros((len(test_files_names), num_classes))
    for model_path in tqdm(models_paths):
        model = torch.load(model_path)
        y_hat_test = make_prediction(model, test_dataset)
        y_hat += y_hat_test / len(models_paths)
    y_hat = np.argmax(y_hat, axis=1)

    # save test predictions
    results = pd.DataFrame({})
    results["wav_id"] = test_files_names
    results["label"] = y_hat
    results.to_csv(path_to_save_results, index=False)
    print("Done!")


if __name__ == "__main__":
    test()

