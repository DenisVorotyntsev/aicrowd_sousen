import os

import torch
from catalyst import dl

from datasets import CustomDataset, collate_fn
from backbones import VGGNet
from models_factory import Supervised1dModel


def train():
    load_to_mem_train = True

    features_stats = {"mean": 86, "std": 22}
    batch_size = 512
    train_batch_size = batch_size
    validation_batch_size = train_batch_size

    path_to_train_data = "./data/external/train/"
    path_to_targets_train = "./data/external/train.csv"
    path_to_val_data = "./data/external/val/"
    path_to_targets_val = "./data/external/val.csv"
    path_to_save_model = "./models/model.pt"

    n_epochs = 2
    es_rounds = 35
    lr = 0.001

    backbone_output_dim = 1024
    backbone = VGGNet()
    model = Supervised1dModel(backbone=backbone, backbone_output_dim=backbone_output_dim, num_classes=3)

    n_workers = os.cpu_count()

    # define train, val and test datasets
    train_dataset = CustomDataset(
        path_to_data=path_to_train_data,
        files_to_use=None,
        load_data_to_mem=load_to_mem_train,
        path_to_targets=path_to_targets_train,
        features_stats=features_stats,
        mode="train"
    )

    val_dataset = CustomDataset(
        path_to_data=path_to_val_data,
        files_to_use=None,
        load_data_to_mem=load_to_mem_train,
        path_to_targets=path_to_targets_val,
        features_stats=features_stats,
        mode="val"
    )

    # define train, val and test loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                               collate_fn=collate_fn,
                                               num_workers=n_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=validation_batch_size,
                                             collate_fn=collate_fn,
                                             num_workers=n_workers, shuffle=False)

    # train
    runner = dl.SupervisedRunner()
    criterion = torch.nn.CrossEntropyLoss()
    callbacks = [
        dl.F1ScoreCallback(),
        dl.EarlyStoppingCallback(patience=es_rounds, minimize=True)
    ]

    print("\n\n")
    print("Main training")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, verbose=True)
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders={"train": train_loader, "valid": val_loader},
        num_epochs=n_epochs,
        callbacks=callbacks,
        logdir="./logdir/",
        load_best_on_end=True,
        main_metric="f1_score",
        minimize_metric=False,
        fp16=True,
        verbose=True
    )

    # save trained model
    torch.save(model, path_to_save_model)


if __name__ == "__main__":
    train()
