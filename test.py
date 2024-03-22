"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import sys
sys.path.append('/home/aistudio/external-libraries')

import lightning.pytorch as pl
from sconf import Config
from torch.utils.data import DataLoader

from dataset import NilmDataset
from model import AadaNet


def test(set_name, house, app_name, data_dir, app_alias, app_threshs, batch_size):
    """
    Train a Nougat model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for training.
    """
    pl.seed_everything(42, workers=True)
    NilmDataset({set_name: [house]}, [app_name], data_dir, app_alias, app_threshs)

    test_set = NilmDataset(config, train=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=18)
    aada = AadaNet.load_from_checkpoint('checkpoinys/aada.ckpt')
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
    )
    trainer.test(aada, test_loader)


if __name__ == "__main__":
    config = Config('config.yaml')
    for set_name, houses in config.test_houses:
        for house in houses:
            for app_name in config.test_apps:
                test(set_name, house, app_name,config.data_dir, config.app_alias, config.app_threshs, config.batch_size)


# examples
# app names/alias