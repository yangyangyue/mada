"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import sys
sys.path.append('/home/aistudio/external-libraries')

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from sconf import Config
from torch.utils.data import DataLoader, random_split

from dataset import NilmDataset
from model import AadaNet


def train(config):
    """
    Train a Nougat model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for training.
    """
    pl.seed_everything(42, workers=True)

    train_set = NilmDataset(config.train_houses, config.train_apps, config.data_dir, config.app_alias, config.app_threshs)
    train_set, val_set = random_split(train_set, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=18)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, num_workers=18)
    aada = AadaNet(config.inplates, config.midplates, config.n_heads, config.dropout, config.n_layers)
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='aada',
        save_last=True
    )
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=config.max_epochs,
        log_every_n_steps=15,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(aada, train_loader, val_loader)


if __name__ == "__main__":
    config = Config('config.yaml')
    train(config)


# examples
# app names/alias