"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

from sconf import Config

import lightning.pytorch as pl
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

    train_set = NilmDataset(config)
    train_set, val_set = random_split(train_set, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)
    aada = AadaNet(config.inplates, config.midplates, config.n_heads, config.dropout, config.n_layers)
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=config.max_epochs,
        log_every_n_steps=15,
        precision="bf16-mixed",
    )

    trainer.fit(aada, train_loader, val_loader)


if __name__ == "__main__":
    config = Config('config.yaml')
    train(config)


# examples
# app names/alias