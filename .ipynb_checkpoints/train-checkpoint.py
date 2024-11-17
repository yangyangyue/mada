"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import argparse
import random

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch
from torch.utils.data import ConcatDataset, random_split, Subset

from dataset import *
from lightning_module import NilmDataModule, NilmNet

def pre_train(method, fit_dataset, save_name):
    # 预训练
    train_set, val_set = random_split(fit_dataset, [0.8, 0.2])
    datamodule = NilmDataModule(train_set=train_set, val_set=val_set, bs=256)
    model = NilmNet(method)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', filename=save_name, monitor="val_mae")
    early_stop_callback = EarlyStopping(monitor="val_mae", patience=20)
    trainer = pl.Trainer(devices="auto", accelerator="auto", max_epochs=80, callbacks=[checkpoint_callback, early_stop_callback], log_every_n_steps=10)
    trainer.fit(model, datamodule=datamodule)

def fine_turning(method, turn_dataset, ckpt_name, save_name):
    # 微调 
    train_set, val_set = random_split(turn_dataset, [0.9, 0.1])
    datamodule = NilmDataModule(train_set=train_set, val_set=val_set, bs=256)
    ckpt_files = list(Path('checkpoints').expanduser().glob(ckpt_name))
    model = NilmNet.load_from_checkpoint(ckpt_files[-1], turning=True, method=method)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', filename=save_name, monitor="val_mae")
    early_stop_callback = EarlyStopping(monitor="val_mae", patience=20)
    trainer = pl.Trainer(devices="auto", accelerator="auto", max_epochs=80, callbacks=[checkpoint_callback, early_stop_callback], log_every_n_steps=10)
    trainer.fit(model, datamodule=datamodule)

def train(method, fithouses, turnhouses, pretrain, noweight):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    # 数据
    if pretrain:  fit_datasets = get_houses_sets(fithouses, noweight, 'fit')
    if turnhouses: turn_datasets = get_syn_houses_sets(turnhouses, noweight)
    suffix = 'noweight' if noweight else 'weight'
    if method != 'mada' and method != 'mvae':
        # 单设备依次训练/微调每个设备
        for i, app_abb in enumerate("kmdwf"):
            print(f"train {app_abb} ...", flush=True)
            if pretrain: pre_train(method, fit_datasets[i], f'{method}-{fithouses}-{app_abb}')
            if turnhouses: fine_turning(method, turn_datasets[i], f'{method}-{fithouses}-{app_abb}*', f'{method}-{fithouses}{turnhouses}-{app_abb}') 
    else:
        # 多设备直接训练/微调
        if pretrain: pre_train(method, ConcatDataset(fit_datasets), f'{method}-{fithouses}-{suffix}')
        if turnhouses: fine_turning(method, ConcatDataset(turn_datasets), f'{method}-{fithouses}-{suffix}*', f'{method}-{fithouses}{turnhouses}-{suffix}')

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='mada') # 方法 mada vae s2p s2s
    parser.add_argument('--fithouses', type=str, default='ukdale15') # 指定用于训练的房屋 ukdale15 refit256
    parser.add_argument('--turnhouses', default='ukdale2') # 指定用于微调的房屋 若为空 则不微调 ukdale2 ukdale1 ukdale5
    parser.add_argument('--noweight', action='store_true') # 不使用加权损失 默认使用
    parser.add_argument('--nopretrain', action='store_true') # 是否预训练
    args = parser.parse_args()
    if args.method=='s2p' or args.method=='s2s': vars.WINDOW_SIZE = 599
    if args.method=='s2p': vars.WINDOW_STRIDE = 1
    # train
    train(args.method, args.fithouses,args.turnhouses, not args.nopretrain, args.noweight)
