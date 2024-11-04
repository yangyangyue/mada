"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import sys
from dataset import vars

import lightning as L
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from models.mada import MadaNet
from models.mvae import MvaeNet
from models.vae import VaeNet
from models.s2p import S2pNet
from models.s2s import S2sNet

class NilmNet(L.LightningModule):
    def __init__(self, method, turning = False, save_path = None) -> None:
        super().__init__()
        self.method = method
        self.turning = turning
        self.save_path = save_path
        if method == 'mada':
            self.model = MadaNet()
        elif method == 'vae':
            self.model = VaeNet()
        elif method == 's2p':
            self.model = S2pNet()
        elif method == 's2s':
            self.model = S2sNet()
        elif method == 'mvae':
            self.model = MvaeNet()
        self.x, self.y, self.y_hat, self.thresh, self.losses = [], [], [], [], []
    
    def forward(self, ids, aggs, tags, apps=None, weights=None):
        if self.training: return self.model(ids, aggs, tags, apps, weights)
        else: return self.model(ids, aggs, tags).relu()
        
    def training_step(self, batch, _):
        # aggs, apps: (N, WINDOE_SIZE)
        # tags: (N, 5)
        # ids, weights, threshs, ceils: (N, )
        ids, aggs, apps, tags, weights, threshs, ceils = batch
        loss = self(ids, aggs, tags, apps, weights)
        self.losses.append(loss.item())
        return loss
        
    def on_train_epoch_end(self) -> None:
        self.log('loss', np.mean(self.losses), on_epoch=True, prog_bar=True, logger=True)
        self.losses.clear()
    
    def validation_step(self, batch, _):
        ids, aggs, apps, tags, weights, threshs, ceils = batch
        pred_apps = self(ids, aggs, tags)
        pred_apps[pred_apps < threshs[:, None]] = 0
        # 记录数据 s2p只记录中心点
        mid = vars.WINDOW_SIZE // 2
        self.y.extend([tensor for tensor in pred_apps])
        self.y_hat.extend([tensor[mid:mid+1] if self.method == 's2p' else tensor for tensor in apps])
        self.thresh.extend([thresh for thresh in threshs])
    
    def on_validation_epoch_end(self):
        mae = torch.concat([y-y_hat for y, y_hat in zip(self.y, self.y_hat)]).abs().mean() 
        mae_on = torch.concat([y[y_hat>thresh] - y_hat[y_hat>thresh] for y, y_hat, thresh in zip(self.y, self.y_hat, self.thresh)]).abs().mean() 
        mre_on = torch.concat([(y[y_hat>thresh] - y_hat[y_hat>thresh]).abs() / y_hat[y_hat>thresh] for y, y_hat, thresh in zip(self.y, self.y_hat, self.thresh)]).mean() 
        self.log('val_mae', mae, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae_on', mae_on, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mre_on', mre_on, on_epoch=True, prog_bar=True, logger=True)
        self.y.clear()
        self.y_hat.clear()
        self.thresh.clear()
    
    def test_step(self, batch, _):
        # 每次只测试一个设备
        ids, aggs, apps, tags, weights, threshs, ceils = batch
        pred_apps = self(ids, aggs, tags)
        pred_apps[pred_apps < threshs[:, None]] = 0
        mid = vars.WINDOW_SIZE // 2
        self.x.extend([tensor[mid:mid+1] if self.method == 's2p' else tensor for tensor in aggs])
        self.y.extend([tensor for tensor in pred_apps])
        self.y_hat.extend([tensor[mid:mid+1] if self.method == 's2p' else tensor for tensor in apps])
        self.thresh.extend([thresh for thresh in threshs])

    def on_test_epoch_end(self):
        x, y, y_hat = torch.concat(self.x), torch.concat(self.y), torch.concat(self.y_hat)
        np.savetxt(self.save_path, torch.stack([x, y_hat, y]).cpu().numpy())
        on_status = y_hat > self.thresh[0]
        mae = (y-y_hat).abs().mean()
        mae_on = (y[on_status]-y_hat[on_status]).abs().mean() 
        mre_on = ((y[on_status]-y_hat[on_status]).abs() / y_hat[on_status]).mean() 
        self.x.clear()
        self.y.clear()
        self.y_hat.clear()
        self.thresh.clear()
        self.print2file('test_mae', mae.item(), 'test_mae_on', mae_on.item(), 'test_mre_on', mre_on.item())
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]

    def print2file(self, *args):
        with open('test_results.log', 'a') as f:
            sys.stdout = f
            print(self.save_path.stem, *args)
            sys.stdout = sys.__stdout__

class NilmDataModule(L.LightningDataModule):
    def __init__(self, train_set=None, val_set=None, test_set=None, bs=256):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = bs

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=18)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=18)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=18)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=18)