"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from sconf import Config
import torch

from lightning_module import NilmDataModule, NilmNet


def train(method, config, houses, app_abbs, tags):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    # model and data
    model = NilmNet(method, config)
    data_module = NilmDataModule(houses, app_abbs, config.data_dir)
    # checkpoint and early stopping
    checkpoint_callback = ModelCheckpoint(
    dirpath='~/checkpoints/',
        filename=f'{method}{tags}-{houses}-{app_abbs}' + '-{epoch}',
        monitor="val_mae"
    )
    early_stop_callback = EarlyStopping(monitor="val_mae", patience=10)
    # trainer
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=1000,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, datamodule=data_module, mode='binsearch')
    # do train and validation
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='aada')
    parser.add_argument('--houses', type=str, default='ukdale15')
    parser.add_argument('--apps', type=str, default='k')
    parser.add_argument('--tags', type=str, default='')
    args = parser.parse_args()
    config = Config('config.yaml')
    # train
    train(args.method, config, args.houses, args.apps, args.tags)
