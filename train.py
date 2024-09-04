"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import argparse
import configparser
import random

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch
from torch.utils.data import ConcatDataset, random_split, Subset

from dataset import get_houses_sets
from lightning_module import NilmDataModule, NilmNet

def train(args, config):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    method, houses =args.method, args.houses
    # model and data
    datasets = get_houses_sets(houses, 'fit')
    # balance the number of samples of diiferent appliances
    min_length = min(len(app_set) for app_set in datasets)
    datasets = [Subset(app_set, random.sample(range(len(app_set)), min_length)) for app_set in datasets]
    dataset = ConcatDataset(datasets)
    train_set, val_set = random_split(dataset, [0.8, 0.2])
    datamodule = NilmDataModule(train_set=train_set, val_set=val_set, bs=config.getint('default', 'batch_size'))
    model = NilmNet(method, config)
    checkpoint_callback = ModelCheckpoint(dirpath='~/checkpoints/', filename=f'{method}-{houses}' + '-{epoch}', monitor="val_mae")
    early_stop_callback = EarlyStopping(monitor="val_mae", patience=config.getint('default', 'patience'))
    trainer = pl.Trainer(devices="auto", accelerator="auto", max_epochs=100, callbacks=[checkpoint_callback, early_stop_callback], log_every_n_steps=10)
    trainer.fit(model, datamodule=datamodule)
    return

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='aada')
    parser.add_argument('--houses', type=str, default='ukdale15')
    args = parser.parse_args()
    # config
    config = configparser.ConfigParser()
    config.read(f'config.ini')
    # train
    train(args, config)
