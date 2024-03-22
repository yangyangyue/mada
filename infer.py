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
import numpy as np
import matplotlib.pyplot as plt
import torch

def test(set_name, house, app_name, data_dir, app_alias, app_threshs, batch_size):
    """
    Train a Nougat model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for training.
    """
    pl.seed_everything(42, workers=True)
    device = torch.device('cuda')
    

    test_set = NilmDataset({set_name: [house]}, [app_name], data_dir, app_alias, app_threshs)
    aada = AadaNet.load_from_checkpoint('checkpoints/aada.ckpt', inplates=config.inplates, midplates=config.midplates, n_heads=config.n_heads, dropout=config.dropout, n_layers=config.n_layers)
    aada = aada.to(device)
    torch.set_grad_enabled(False)
    aada.eval()
    for i, (_, example, sample, app) in enumerate(test_set):
        if np.max(app)< 60:
            continue
        example, sample = torch.tensor(example).to(device), torch.tensor(sample).to(device)
        pred_apps = aada(example[None, :], sample[None, :])[0]
        pred_apps[pred_apps < 15] = 0
        plt.figure()
        plt.plot(example.cpu().numpy(), label='example')
        plt.plot(sample.cpu().numpy(), label='sample')
        plt.plot(app, label='app')
        plt.plot(pred_apps.cpu().numpy(), label='pred_app')
        plt.legend()
        plt.savefig(f'case/ukdale_{house}_{app_name}_{i}.png')
        plt.close()


if __name__ == "__main__":
    config = Config('config.yaml')
    for set_name, houses in config.test_houses.items():
        for house in houses:
            for app_name in config.test_apps:
                test(set_name, house, app_name,config.data_dir, config.app_alias, config.app_threshs, config.batch_size)