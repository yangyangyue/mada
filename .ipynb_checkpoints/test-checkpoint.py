"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import argparse
import configparser
from pathlib import Path

import lightning.pytorch as pl
import torch

from dataset import get_houses_sets
from lightning_module import NilmDataModule, NilmNet

def test(method, houses, ckpt, o2o, onehot):
    # init
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    # model and data
    datasets = get_houses_sets(houses, onehot, False, 'test')
    for app_abb, app_set in zip('kmdwf', datasets):
        save_path = Path('results') / f'{method}-{houses}-{app_abb}.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_files = list(Path('~/checkpoints').expanduser().glob(f'{args.ckpt}-{app_abb if o2o else "a"}*'))
        datamodule = NilmDataModule(test_set=app_set, bs=256)
        model = NilmNet.load_from_checkpoint(ckpt_files[0], method=method, save_path=save_path)
        trainer = pl.Trainer(devices="auto", accelerator="auto")
        trainer.test(model, datamodule=datamodule, verbose=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='aada')
    parser.add_argument('--houses', type=str, default='ukdale2')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--o2o', action='store_true')
    parser.add_argument('--onehot', action='store_true')
    args = parser.parse_args()
    test(args.method, args.houses, args.ckpt, args.o2o, args.onehot)
        