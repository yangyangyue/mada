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

from lightning_module import NilmDataModule, NilmNet


def train(method, config, houses, app_names):
    pl.seed_everything(42, workers=True)
    # model and data
    model = NilmNet(method, config)
    data_module = NilmDataModule(houses, app_names, config.data_dir)
    # checkpoint and early stopping
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename=f'{method}-{houses}-{app_names}' + '-{epoch}',
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
    tuner.scale_batch_size(model, datamodule=data_module)
    # do train and validation
    trainer.fit(model, data_module=data_module)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='aada')
    parser.add_argument('--houses', type=str, default='u15')
    parser.add_argument('--apps', type=str, default='k')
    args = parser.parse_args()
    config = Config('config.yaml')
    # train
    train(args.method, config, args.houses, args.apps)
