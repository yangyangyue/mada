"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import argparse

import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from sconf import Config
import torch

from lightning_module import NilmDataModule, NilmNet


def test(method, config, houses, app_abb, ckpt):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')

    model = NilmNet.load_from_checkpoint(f'~/checkpoints/{ckpt}.ckpt', net_name=method, config=config)
    data_module = NilmDataModule(houses, app_abb, config.data_dir)
    trainer = pl.Trainer(devices="auto", accelerator="auto")
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, datamodule=data_module, method='test', mode='binsearch')
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='aada')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--houses', type=str, default='ukdale15')
    parser.add_argument('--apps', type=str, default='k')
    args = parser.parse_args()

    config = Config('config.yaml')
    for app_abb in args.apps:
        test(args.method, config, args.houses, app_abb,  args.ckpt)