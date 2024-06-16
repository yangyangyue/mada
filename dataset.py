"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

""" Tips
1. 如果一个房屋，针对一个appliance，有多个instance，那么需要将他们合并在一起，还是单独分开？
核心问题是同设备不同实例的波形相似而又不完全相同
1）将他们分开会有一个问题，比如当前在训练instance1，那么总线上出现instance2的波形，这时候就会有问题。即总线上有这个设备，但支线上没有。虽然不是一个设备，但实例的波形很相似的，会导致不稳定问题。
2）将他们合并在一起，但是在我的迁移方案下，只记录一个instance的示例，从逻辑上来说就不应该推出别的instance的波形
解决方案：每个实例采集一个示例，然后合并在一起。
"""

WINDOW_SIZE = 1024
WINDOW_STRIDE = 256

threshs ={"k": 2000, "m": 200, "d": 20, "w": 1200, "f": 50}
ceils = {"k": 3100, "m": 3000, "d": 2500, "w": 2500, "f": 300}
ukdale_channels = {
    'k': {1: [10], 2: [8], 5: [18]},
    'm': {1: [13], 2: [15], 5: [23]},
    'd': {1: [6], 2: [13], 5: [22]},
    'w': {1: [5], 2: [12], 5: [24]},
    'f': {1: [12], 2: [14], 5: [19]}
}
# refit_channels = {
#     'k':{2: [8], 3: [9], 4: [9], 5: [8], 6: [7], 7: [9], 8: [9], 9: [7], 11: [7], 12: [6], 13: [9], 17: [8], 19: [5], 20: [9], 21: [7]},
#     'm':{2: [5], 3: [8], 4: [8], 5: [7], 6: [6], 8: [8], 9: [6], 10: [8], 11: [6], 12: [5], 13:[7, 8], 15: [7], 17: [7], 18: [9], 19: [4], 20: [8]},
#     'd':{1: [6], 2: [3], 3: [5], 5: [4], 6: [3], 7: [6], 9: [4], 10:[6], 11: [4], 13: [4], 15: [4], 16: [6], 18: [6], 20: [5], 21: [4]},
#     'w':{1: [4, 5], 2: [2], 3: [6], 4: [4, 5], 5:[3], 6: [2], 7: [5], 8: [3, 4], 9: [2, 3], 10: [5], 11: [3], 13: [3], 15: [3], 16: [5], 
#          17: [4], 18:[4,5], 19:[2], 20: [4], 21: [3]},
#     'f':{1: [1, 2, 3], 2:[1], 3: [2, 3], 4:[1, 2, 3], 5: [1], 6: [1], 7:[1, 2, 3], 8: [1, 2], 9: [1], 10: [4], 11: [1, 2], 12:[1], 13: [2], 
#          15: [1], 16: [1, 2], 17: [1, 2], 18: [1, 2, 3], 19: [1], 20: [1, 2], 21: [1]}   
# }

refit_channels = {
    'k':{2: [8], 5: [8], 6: [7]},
    'm':{2: [5], 5: [7], 6: [6]},
    'd':{2: [3], 5: [4], 6: [3]},
    'w':{2: [2], 5: [3], 6: [2]},
    'f':{2: [1], 5: [1], 6: [1]}
}

def read(data_dir, set_name, house_id, app_abb=None, channel=None):
    """ read a dataframe of a specific appliance or mains """
    if set_name == 'ukdale':
        if not app_abb:
            channel = 1
        path = data_dir / set_name / f'house_{house_id}' / f'channel_{channel}.dat'
        df = pd.read_csv(path, sep=" ", header=None).iloc[:, :2]
    elif set_name == 'refit':
        path = data_dir / set_name / f'CLEAN_House{house_id}.csv'
        df = pd.read_csv(path)
        column = 'Aggregate' if not app_abb else f'Appliance{channel}'
        df = df[['Unix', column]]
    df.columns = ["stamp", "power"]
    df.sort_values(by="stamp")
    df['stamp'] = pd.to_datetime(df['stamp'], unit='s')
    df = df.set_index('stamp')
    df = df.resample('6s').mean().ffill(limit=30).dropna()
    df[df < (threshs[app_abb] if app_abb else 15)] = 0
    df = df.clip(lower=0, upper=6000 if not app_abb else ceils[app_abb])
    return df


class NilmDataset(Dataset):
    """ designed for one appliance in one house """
    def __init__(self, dir, set_name, house_id, app_abb, stage) -> None:
        super().__init__()
        print(f'{set_name}-{house_id}-{app_abb} is loading...')
        self.dir = dir
        self.set_name = set_name
        self.house_id = house_id
        self.app_abb = app_abb
        self.app_thresh = threshs[app_abb]
        self.app_ceil = ceils[app_abb]
        self.samples, self.apps = self.load_data()
        if len(self.samples) < WINDOW_SIZE:
            self.samples, self.apps, self.example = [], [], None
        else:
            self.samples = np.copy(np.lib.stride_tricks.sliding_window_view(self.samples, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32)
            self.apps = np.copy(np.lib.stride_tricks.sliding_window_view(self.apps, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32)
            self.example = np.load(Path('examples') / f'{set_name}{house_id}-{app_abb}.npy')[0]
            self.example = np.clip(self.example, 0, self.app_ceil)
        if stage != 'fit':
            return 
        # for ukdale house_1, only select 15% of data
        if self.set_name == 'ukdale' and self.house_id == 1:
            num = len(self.samples)
            ind = np.random.permutation(num)
            select_ids = ind[: int(0.15 * num)]
            self.samples = self.samples[select_ids]
            self.apps = self.apps[select_ids]

        # balance the number of samples
        # pos_idx = np.nonzero(np.any(self.apps >= self.app_thresh, axis=1))[0]
        # neg_idx = np.nonzero(np.any(self.apps < self.app_thresh, axis=1))[0]
        # if 1 * len(pos_idx) < len(neg_idx):
        #     neg_idx = np.random.choice(neg_idx, 1 * len(pos_idx), replace=False)
        #     self.samples = np.concatenate([self.samples[pos_idx], self.samples[neg_idx]])
        #     self.apps = np.concatenate([self.apps[pos_idx], self.apps[neg_idx]])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """ the shape of each part is  (window_size, ) """
        return self.samples[idx], self.apps[idx], self.example, self.app_thresh, self.app_ceil
    
    def load_data(self):
        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / hashlib.sha256(f'{self.set_name}{self.house_id}-{self.app_abb}'.encode()).hexdigest()
        if temp_path.exists():
            powers = np.load(temp_path)
        else:
            aggs = read(self.dir, self.set_name, self.house_id)
            channels = ukdale_channels[self.app_abb][self.house_id] if self.set_name == 'ukdale' else refit_channels[self.app_abb][self.house_id]
            apps = read(self.dir, self.set_name, self.house_id, self.app_abb, channels[0])
            powers = pd.merge(aggs, apps, on='stamp').to_numpy(dtype=np.float32)
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(temp_path, powers)
        return powers[:, 0], powers[:, 1]