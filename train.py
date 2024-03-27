"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import argparse
import sys
sys.path.append('/home/aistudio/external-libraries')

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sconf import Config
from torch.utils.data import DataLoader, random_split

from dataset import NilmDataset
from lightning_module import NilmNet


def train(method, config, houses, app_names):
    """
    Train a Nougat model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for training.
    """
    pl.seed_everything(42, workers=True)

    # data
    train_set = NilmDataset(houses, app_names, config.data_dir, config.alias, config.threshs)
    train_set, val_set = random_split(train_set, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=18)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, num_workers=18)

    # model
    model = NilmNet(method, config)
    
    # checkpoint
    filename = method
    for set_name, houses_in_set in houses.items():
        filename += f'-{set_name[0]}'
        for house in houses_in_set:
            filename += f'{house[-1]}'
    filename += '-'
    for app_name in app_names:
        filename += f'{app_name[0]}'
    filename += '-{epoch}'
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename=filename,
        monitor="val_mae"
    )

    # early stopping
    early_stop_callback = EarlyStopping(monitor="val_mae", patience=10)
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=config.max_epochs,
        log_every_n_steps=15,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(model, train_loader, val_loader)

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='aada')
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
    method = args.method
    houses = {ch2set[ele[0]]: [f'house_{id}' for id in ele[1:]]
              for ele in args.houses.split('-')}
    app_names = [ch2app[ele] for ele in args.apps]
    train(method, config, houses, app_names)