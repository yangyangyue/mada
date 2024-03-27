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

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='aada')
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--houses', type=str, default='u15')
parser.add_argument('--apps', type=str, default='k')
args = parser.parse_args()

ch2set = {
    'u': 'ukdale',
    'd': 'redd',
    'f': 'refit'
}
ch2app = {
    'k': 'kettle',
    'm': 'microwave',
    'd': 'dishwasher',
    'w': 'washing_machine',
    'f': 'fridge'
}


if __name__ == "__main__":
    config = Config('config.yaml')
    method, ckpt = args.method, args.ckpt
    houses = {ch2set[ele[0]]: [f'house_{id}' for id in ele[1:]]
              for ele in args.houses.split('-')}
    app_names = [ch2app[ele] for ele in args.apps]
    for set_name, houses_in_set in houses.items():
        for house in houses_in_set:
            for app_name in app_names:
                test(method, ckpt, set_name, house, app_name,config.data_dir, config.alias, config.threshs, config.batch_size)