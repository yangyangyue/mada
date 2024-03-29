
from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader

from dataset import ReddDataset, UkdaleDataset, random_split
from models.avae import AvaeNet

import lightning as L
import torch
from torch.optim.lr_scheduler import LambdaLR

from models.aada import AadaNet
from models.vae import VaeNet


WINDOW_SIZE = 1024
WINDOW_STRIDE = 256   

abb2name = {
    'k': 'kettle',
    'm': 'microwave',
    'd': 'dishwasher',
    'w': 'washing_machine',
    'f': 'fridge'
}

alias = {
    "kettle": ["kettle"],
    "microwave": ["microwave"],
    "dishwasher": ["dishwasher", "dish_washer", "dishwaser"],
    "washing_machine": ["washing_machine", "washer_dryer"],
    "fridge": ["fridge", "fridge_freezer", "refrigerator"],
  }

threshs ={
    "kettle": 2000,
    "fridge": 50,
    "washing_machine": 20,
    "microwave": 200,
    "dishwasher": 10,
}

ceils = {
    "kettle": 3100,
    "fridge": 300,
    "washing_machine": 2500,
    "microwave": 3000,
    "dishwasher": 2500,
}


class NilmNet(L.LightningModule):
    def __init__(self, net_name, config) -> None:
        super().__init__()
        self.batch_size = None
        self.config = config
        if net_name == 'aada':
            self.model = AadaNet(
                self_attention=config.self_attention, 
                channels=config.channels,
                dropout=config.dropout, 
                n_heads=config.n_heads, 
                mid_channels=config.mid_channels, 
                use_ins=False, 
                n_layers=config.n_layers,
                variation = config.variation
            )
        elif net_name == 'vae':
            self.model = VaeNet()
        elif net_name == 'avae':
            self.model = AvaeNet()
        self.y = []
        self.y_hat = []
        self.thresh = []
    
    def forward(self, examples, samples, gt_apps=None):
        return self.model(examples, samples, gt_apps)
    
    
    def training_step(self, batch, _):
        # examples | samples | gt_apps: (N, WINDOE_SIZE)
        _, examples, samples, gt_apps = batch
        loss = self(examples, samples, gt_apps)
        self.log('loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, _):
        # tags: (N, 3)
        # examples | samples | gt_apps: (N, WINDOE_SIZE)
        threshs, examples, samples, gt_apps = batch
        pred_apps = self(examples, samples)
        pred_apps[pred_apps < 15] = 0
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
        threshs, examples, samples, gt_apps = batch
        pred_apps = self(examples, samples)
        pred_apps[pred_apps < 15] = 0
        self.y.extend([tensor for tensor in pred_apps])
        self.y_hat.extend([tensor for tensor in gt_apps])
        self.thresh.extend([thresh for thresh in threshs])

    def on_test_epoch_end(self):
        device = self.thresh[0].device
        y = reconstruct(self.y).to(device)
        y_hat = reconstruct(self.y_hat).to(device)
        mae = (y-y_hat).abs().mean()
        on_status = y_hat > self.thresh[0]
        mae_on = (y[on_status]-y_hat[on_status]).abs().mean() 
        mre_on = ((y[on_status]-y_hat[on_status]).abs() / y_hat[on_status]).mean() 
        self.log('test_mae', mae, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mae_on', mae_on, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mre_on', mre_on, on_epoch=True, prog_bar=True, logger=True)
        self.y.clear()
        self.y_hat.clear()
        self.thresh.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = {
            "scheduler": self.exponential_scheduler(
                optimizer,
                200,
                self.config.lr,
                self.config.min_lr
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler]
    
    @staticmethod
    def exponential_scheduler(optimizer, warmup_steps, lr, min_lr=1e-5, gamma=0.9999):
        def lr_lambda(x):
            if x > warmup_steps:
                if lr * gamma ** (x - warmup_steps) > min_lr:
                    return gamma ** (x - warmup_steps)
                else:
                    return min_lr / lr
            else:
                return x / warmup_steps

        return LambdaLR(optimizer, lr_lambda=lr_lambda)


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
    def __init__(self, houses, app_names,  data_dir, batch_size = 32):
        super().__init__()
        self.houses = houses
        self.app_names = app_names
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage):
        houses = {ele[0]: [f'house_{id}' for id in ele[1:]] for ele in self.houses.split('_')}
        app_names = [abb2name[ele] for ele in self.apps]
        datasets = []
        # build dataset for appliances in ukdale
        if 'u' in self.houses:
            dir = Path(self.data_dir) / 'ukdale'
            datasets += [UkdaleDataset(dir,  house, app_name, alias[app_name], threshs[app_name]) 
                            for house in houses['u'] for app_name in app_names]
        # build dataset for appliances in redd
        if 'd' in self.houses:
            dir = Path(self.data_dir) / 'redd'
            datasets += [ReddDataset(dir, house, app_name, alias[app_name], threshs[app_name]) 
                            for house in houses['d'] for app_name in app_names]
        dataset = ConcatDataset(datasets)

        if stage == 'fit':
            self.train_set, self.val_set = random_split(dataset, [0.8, 0.2])
        else:
            self.test_set = self.dataset

    def train_dataloader(self):
        print('batch size:', self.batch_size)
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=18)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=18)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=18)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=18)