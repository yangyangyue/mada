"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""
import sys
sys.path.append('..')
from dataset import vars
import argparse
import configparser
from pathlib import Path

import lightning.pytorch as pl
import torch

from dataset import get_houses_sets
from lightning_module import NilmDataModule, NilmNet

def test(method, test_house, fit_houses, fine_houses, o2o):
    # init
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    # model and data
    set_name, house_id = test_house[:-1], test_house[-1]
    datasets = get_house_sets(set_name, house_id, 'test')
    for app_abb, app_set in zip('kmdwf', datasets):
        save_path = Path('results') / f'{method}-{houses}-{app_abb}.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_files = list(Path('~/checkpoints').expanduser().glob(f'{args.ckpt}-{app_abb if o2o else "a"}'))
        datamodule = NilmDataModule(test_set=app_set, bs=256)
        model = NilmNet.load_from_checkpoint(ckpt_files[-1], method=method, save_path=save_path)
        trainer = pl.Trainer(devices="auto", accelerator="auto")
        trainer.test(model, datamodule=datamodule, verbose=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='aada')
    parser.add_argument('--test_house', type=str, default='ukdale2')
    parser.add_argument('--fit_houses', type=str, default='ukdale15')
    parser.add_argument('--fine_houses', type=str, default='ukdale2')
    parser.add_argument('--o2o', action='store_true')
    args = parser.parse_args()
    vars.WINDOW_STRIDE = 1 if method=='s2p' else 1024
    test(args.method, args.test_house, args.fit_houses, args.fine_houses, args.o2o)
        