"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import random
import re

import lightning as L
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader, random_split, Subset

from dataset import NilmDataset
from models.aada import AadaNet
from models.acvae import AcvaeNet
from models.avae import AvaeNet
from models.avae2t import Avae2tNet
from models.vae import VaeNet

WINDOW_SIZE = 1024
WINDOW_STRIDE = 256   


class NilmNet(L.LightningModule):
    def __init__(self, net_name, config, save_path = None) -> None:
        super().__init__()
        self.batch_size = None
        self.lr = config.getfloat('default', 'lr')
        self.min_lr = config.getfloat('default', 'min_lr')
        self.save_path = save_path
        if net_name == 'aada':
            sec = config['aada']
            self.model = AadaNet(
                self_attention=sec.getboolean('self_attention'), 
                channels=sec.getint('channels'),
                dropout=sec.getfloat('dropout'),
                n_heads=sec.getint('n_heads'),
                mid_channels=sec.getint('mid_channels'),
                use_ins=False, 
                n_layers=sec.getint('n_layers'),
                variation = sec.getboolean('variation')
            )
        elif net_name == 'vae':
            self.model = VaeNet()
        elif net_name == 'avae':
            self.model = AvaeNet()
        elif net_name == 'avae2t':
            self.model = Avae2tNet()
        elif net_name == 'acave':
            self.model = AcvaeNet()
        self.x = []
        self.y = []
        self.y_hat = []
        self.thresh = []
        self.losses = []
    
    def forward(self, examples, samples, gt_apps=None):
        return self.model(examples, samples, gt_apps)
    
    
    def training_step(self, batch, _):
        # examples | samples | gt_apps: (N, WINDOE_SIZE), threshs | ceils: (N, )
        samples, gt_apps, examples, threshs, ceils = batch
        loss = self(examples, samples, gt_apps)
        self.losses.append(loss.item())
        return loss
    def on_train_epoch_end(self) -> None:
        self.log('loss', np.mean(self.losses), on_epoch=True, prog_bar=True, logger=True)
        self.losses.clear()
    
    def validation_step(self, batch, _):
        # tags: (N, 3)
        # examples | samples | gt_apps: (N, WINDOE_SIZE)
        samples, gt_apps, examples, threshs, ceils = batch
        pred_apps = self(examples, samples)
        pred_apps[pred_apps < threshs[:, None]] = 0
        self.y.extend([tensor for tensor in pred_apps])
        self.y_hat.extend([tensor for tensor in gt_apps])
        self.thresh.extend([thresh for thresh in threshs])
    
    def on_validation_epoch_end(self):
        mae = torch.concat([y-y_hat for y, y_hat in zip(self.y, self.y_hat)]).abs().mean() 
        mae_on = torch.concat([y[y_hat>thresh] - y_hat[y_hat>thresh] for y, y_hat, thresh in zip(self.y, self.y_hat, self.thresh)]).abs().mean() 
        mre_on = torch.concat([(y[y_hat>thresh] - y_hat[y_hat>thresh]).abs() / y_hat[y_hat>thresh]
                                for y, y_hat, thresh in zip(self.y, self.y_hat, self.thresh)]).mean() 
        self.log('val_mae', mae, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae_on', mae_on, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mre_on', mre_on, on_epoch=True, prog_bar=True, logger=True)
        self.y.clear()
        self.y_hat.clear()
        self.thresh.clear()
    
    def test_step(self, batch, _):
        samples, gt_apps, examples, threshs, ceils = batch
        pred_apps = self(examples, samples)
        pred_apps[pred_apps < threshs[:, None]] = 0
        self.x.extend([tensor for tensor in samples])
        self.y.extend([tensor for tensor in pred_apps])
        self.y_hat.extend([tensor for tensor in gt_apps])
        self.thresh.extend([thresh for thresh in threshs])

    def on_test_epoch_end(self):
        device = self.thresh[0].device
        x = reconstruct(self.x).to(device)
        y = reconstruct(self.y).to(device)
        y_hat = reconstruct(self.y_hat).to(device)
        np.savetxt(self.save_path, torch.stack([x, y_hat, y]).cpu().numpy())
        mae = (y-y_hat).abs().mean()
        on_status = y_hat > self.thresh[0]
        mae_on = (y[on_status]-y_hat[on_status]).abs().mean() 
        mre_on = ((y[on_status]-y_hat[on_status]).abs() / y_hat[on_status]).mean() 
        self.x.clear()
        self.y.clear()
        self.y_hat.clear()
        self.thresh.clear()
        print('test_mae', mae, 'test_mae_on', mae_on, 'test_mre_on', mre_on)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())
        # scheduler = {
        #     "scheduler": self.exponential_scheduler(
        #         optimizer,
        #         200,
        #         self.lr,
        #         self.min_lr
        #     ),
        #     "name": "learning_rate",
        #     "interval": "step",
        #     "frequency": 1
        # }
        # return [optimizer], [scheduler]
    
    # @staticmethod
    # def exponential_scheduler(optimizer, warmup_steps, lr, min_lr=1e-5, gamma=0.9999):
    #     def lr_lambda(x):
    #         if x > warmup_steps:
    #             if lr * gamma ** (x - warmup_steps) > min_lr:
    #                 return gamma ** (x - warmup_steps)
    #             else:
    #                 return min_lr / lr
    #         else:
    #             return x / warmup_steps

    #     return LambdaLR(optimizer, lr_lambda=lr_lambda)


def reconstruct(y):
    n = len(y)
    length = WINDOW_SIZE + (n - 1) * WINDOW_STRIDE 
    depth = WINDOW_SIZE // WINDOW_STRIDE
    out = torch.full([length, depth], float('nan'))
    for i, cur in enumerate(y):
        start = i * WINDOW_STRIDE
        d = i % depth
        out[start: start+WINDOW_SIZE, d] = cur
    out = torch.nanmedian(out, dim=-1).values
    return out

class NilmDataModule(L.LightningDataModule):
    def __init__(self, houses, app_abbs, data_dir, batch_size=64):
        super().__init__()
        self.houses = houses
        self.app_abbs = app_abbs
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage):
        if stage != 'fit':
            # just load dataset of one appliance in one house
            match = re.match(r'^(\D+)(\d+)$', self.houses)
            set_name, house_ids = match.groups()
            self.test_set = NilmDataset(Path(self.data_dir), set_name, int(house_ids), self.app_abbs, stage)
            return
        with ThreadPoolExecutor(16) as executor:
            future_map = {}
            for app_abb in self.app_abbs:
                future_map[app_abb] = []
                for houses_in_set in self.houses.split('-'):
                    match = re.match(r'^(\D+)(\d+)$', houses_in_set)
                    set_name, house_ids = match.groups()
                    future_map[app_abb] += [executor.submit(NilmDataset, Path(self.data_dir), set_name, int(house_id), app_abb, stage) for house_id in house_ids]
        # get NilmDatasets of each appliance
        datasets = [ConcatDataset([future.result() for future in as_completed(future_map[app_abb])]) for app_abb in self.app_abbs]
        # balance the number of samples of diiferent appliances
        min_length = min(len(dataset) for dataset in datasets)
        max_length = int(min_length * 1.5)
        balanced_datasets = []
        for dataset in datasets:
            if len(dataset) > max_length:
                indices = random.sample(range(len(dataset)), max_length)
                balanced_dataset = Subset(dataset, indices)
            else:
                balanced_dataset = dataset
            balanced_datasets.append(balanced_dataset)
        dataset = ConcatDataset(datasets)
        self.train_set, self.val_set = random_split(dataset, [0.8, 0.2])
            

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=18)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=18)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=18)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=18)