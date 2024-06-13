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

from lightning_module import NilmDataModule, NilmNet


def test(method, houses, app_abb, ckpt, config):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    save_path = Path('results') / f'{method}-{houses}-{app_abb}.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_files = list(Path('~/checkpoints').expanduser().glob(f'{ckpt}*'))
    model = NilmNet.load_from_checkpoint(ckpt_files[0], net_name=method, sec=config['aada'], save_path=save_path)
    datamodule = NilmDataModule(houses, app_abb, config.get('default', 'data_dir'), batch_size=32)
    trainer = pl.Trainer(devices="auto", accelerator="auto")
    trainer.test(model, datamodule=datamodule, verbose=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='aada')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--houses', type=str, default='ukdale15')
    parser.add_argument('--apps', type=str, default='k')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(f'config.ini')
    for app_abb in args.apps:
        test(args.method, args.houses, app_abb,  args.ckpt, config)
        