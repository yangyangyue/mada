"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import argparse
import sys

from lightning_module import NilmNet
sys.path.append('/home/aistudio/external-libraries')

import lightning.pytorch as pl
from sconf import Config
from torch.utils.data import DataLoader

from dataset import NilmDataset
from models.aada import AadaNet

houses = { "ukdale": ["house_2"] }
app_names = ["kettle", "microwave", "dishwasher", "washing_machine", "fridge"]

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='aada')
parser.add_argument('--ckpt', type=str, required=True)
args = parser.parse_args()


def test(method, ckpt, set_name, house, app_name, data_dir, app_alias, app_threshs, batch_size):
    """
    Train a Nougat model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for training.
    """
    pl.seed_everything(42, workers=True)
    

    test_set = NilmDataset({set_name: [house]}, [app_name], data_dir, app_alias, app_threshs)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=18)

    model = NilmNet.load_from_checkpoint(f'checkpoints/{ckpt}.ckpt', net_name=method, config=config)
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
    )
    trainer.test(model, test_loader)


if __name__ == "__main__":
    config = Config('config.yaml')
    method = args.method
    ckpt = args.ckpt
    for set_name, houses_in_set in houses.items():
        for house in houses_in_set:
            for app_name in app_names:
                test(method, ckpt, set_name, house, app_name,config.data_dir, config.alias, config.threshs, config.batch_size)