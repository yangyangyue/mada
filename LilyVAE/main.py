"""
the start point

written by lily
email: lily231147@gmail.com
"""

import random
from math import inf
from typing import Union

import numpy as np
import torch
from rich.progress import track
from torch.optim.lr_scheduler import StepLR

from .vaenet import VaeNet
from util.config import UkdaleConfig, ReddConfig
from util.load_data import get_loaders
from util.metric import Metric


def train(app_name: str, config: Union[ReddConfig, UkdaleConfig]):
    """ train and val for app specified by app_name. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_path = f"./LilyVAE/weights/{app_name}.pth"
    # data loader
    train_loader, val_loader, (main_mean, main_std, app_mean, app_std) = get_loaders(app_name, config)
    # model
    if config.method == 'vae':
        model = VaeNet(main_mean, main_std, app_mean, app_std).to(device)
    else:
        raise ValueError("method must in (vae, s2p)")

    # optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=config.lr, betas=config.optimizer_args['adam']['betas'])
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=config.lr)
    else:
        raise ValueError("optimizer must in (adam, adamw, sgd)")
    scheduler = StepLR(optimizer, step_size=config.scheduler['step_size'], gamma=config.scheduler['gamma'])

    # train and val
    mae_best = inf
    for epoch in range(0, config.n_epoch):
        model.train()
        for aggs, apps, status in track(train_loader, description=f"Epoch: {epoch}  train"):
            aggs, apps, status = aggs.to(device), apps.to(device), status.to(device)
            loss = model(aggs, apps, status)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        metric = Metric(threshold:=config.threshold[app_name], config.window_size, config.window_stride)
        for aggs, apps, status in track(val_loader, description=f"Epoch: {epoch}  test"):
            aggs, apps, status = aggs.to(device), apps.to(device), status.to(device)
            apps_pred = model(aggs)
            apps_pred[ apps_pred< threshold]=0
            metric.add(apps, apps_pred)
        if metric.get_metrics()[-2] < mae_best:
            mae_best = metric.get_metrics()[-2]
            save_files = {'epoch': epoch, 'mae_best': mae_best,
                          'model': model.state_dict(), 'stat': (main_mean, main_std, app_mean, app_std),
                          'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(save_files, weight_path)
        print(f"epoch: {epoch}, metrics: {metric.get_metrics()}")


def main():
    """ iterate each appliance of ukdale and redd"""
    # ukdale
    config = UkdaleConfig()
    for app_name in config.app_names:
        train(app_name, config)
    # redd
    config = ReddConfig()
    for app_name in config.app_names:
        train(app_name, config)
