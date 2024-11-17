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

from dataset import get_house_sets
from lightning_module import NilmDataModule, NilmNet

def test(method, test_house, fit_houses, turn_houses, noweight):
    # init
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    # model and data
    set_name, house_id = test_house[:-1], test_house[-1]
    datasets = get_house_sets(set_name, int(house_id), False, test)
    for app_abb, app_set in zip('kmdwf', datasets):
        if method != 'mada' and method != 'mvae': suffix = app_abb
        elif noweight: suffix = 'noweight'
        else: suffix = 'weight'
        save_path = Path('results') / f'{'manw' if method == 'mada' and noweight else method}-{fit_houses}{turn_houses}-{app_abb}.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        datamodule = NilmDataModule(test_set=app_set, bs=256)
        ckpt_files = list(Path('checkpoints').expanduser().glob(f"{method}-{fit_houses}{turn_houses}-{suffix}*"))
        model = NilmNet.load_from_checkpoint(ckpt_files[-1], method=method, save_path=save_path)
        trainer = pl.Trainer(devices="auto", accelerator="auto")
        trainer.test(model, datamodule=datamodule, verbose=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='mada')
    parser.add_argument('--testhouse', type=str, default='ukdale2')
    parser.add_argument('--fithouses', type=str, default='ukdale15')
    parser.add_argument('--turnhouses', type=str, default='ukdale2')
    parser.add_argument('--noweight', action='store_true')
    args = parser.parse_args()
    if args.method=='s2p' or args.method=='s2s': vars.WINDOW_SIZE = 599
    vars.WINDOW_STRIDE = 1 if args.method=='s2p' else vars.WINDOW_SIZE
    test(args.method, args.testhouse, args.fithouses, args.turnhouses, args.noweight)
        