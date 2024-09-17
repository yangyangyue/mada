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

def do_train(app_abb, method, houses, tag, dataset):
    train_set, val_set = random_split(dataset, [0.8, 0.2])
    datamodule = NilmDataModule(train_set=train_set, val_set=val_set, bs=256)
    model = NilmNet(method)
    checkpoint_callback = ModelCheckpoint(dirpath='~/checkpoints/', filename=f'{method}-{tag}-{houses}' + '-{epoch}-' + app_abb, monitor="val_mae")
    early_stop_callback = EarlyStopping(monitor="val_mae", patience=20)
    trainer = pl.Trainer(devices="auto", accelerator="auto", max_epochs=80, min_epochs=30, callbacks=[checkpoint_callback, early_stop_callback], log_every_n_steps=10)
    trainer.fit(model, datamodule=datamodule)

def train(method, houses, tag, o2o, onehot, noweight):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    # data
    datasets = get_houses_sets(houses, onehot, noweight, 'fit')
    if not o2o:
        # 不进行正负样本平衡的话，设备间样本平衡也没有意义了
        # min_length = min(len(app_set) for app_set in datasets)
        # datasets = [Subset(app_set, random.sample(range(len(app_set)), min_length)) for app_set in datasets]
        dataset = ConcatDataset(datasets)
        do_train('a', method, houses, tag, dataset)
    else:
        for app_abb, dataset in zip("kmdwf", datasets):
            print(f"train {app_abb} ...", flush=True)
            do_train(app_abb, method, houses, tag)

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='aada')
    parser.add_argument('--houses', type=str, default='ukdale15')
    parser.add_argument('--tag', type=str, default='stable') # 保存的ckpt的tag
    parser.add_argument('--o2o', action='store_true')  # 指定是单模型单设备还是单模型多设备
    parser.add_argument('--onehot', action='store_true') # 使用onehot作为负荷印记
    parser.add_argument('--noweight', action='store_true') # 不使用加权损失
    args = parser.parse_args()
    # train
    train(args.method, args.houses, args.tag, args.o2o, args.onehot, args.noweight)
