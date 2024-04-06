"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import argparse
import configparser

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch

from lightning_module import NilmDataModule, NilmNet

def train(args, config):
    # init
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    method, houses, app_abbs, tags =args.method, args.houses, args.apps, args.tags
    # model and data
    model = NilmNet(args.method, config)
    data_module = NilmDataModule(houses, app_abbs, config.get('default', 'data_dir'), batch_size=64)
    # checkpoint and early stopping
    checkpoint_callback = ModelCheckpoint(
        dirpath='~/checkpoints/',
        filename=f'{method}{tags}-{houses}-{app_abbs}' + '-{epoch}',
        monitor="val_mae"
    )
    early_stop_callback = EarlyStopping(monitor="val_mae", patience=20)
    # trainer
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10
    )
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
    # config
    config = configparser.ConfigParser()
    config.read('config.ini')
    # train
    train(args, config)
