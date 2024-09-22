"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import hashlib
from math import exp
from pathlib import Path
import random
import re
import tempfile

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, ConcatDataset

DIR = Path('/root/autodl-tmp/nilm_lf')
ids = {"k": 0, "m": 1, "d": 2, "w": 3, "f": 4}
threshs = {"k": 2000, "m": 200, "d": 20, "w": 1200, "f": 50}
ceils = {"k": 3100, "m": 3000, "d": 2500, "w": 2500, "f": 300}
names = ['kettle', 'microwave', 'dishwasher', 'washing_machine', 'fridge']
# 计算各电器的损失权重：[1.00, 1.03, 1.24, 1.24, 10.33]
weights = 1 / np.array(list(ceils.values()))
weights = weights / np.min(weights)
# onehot
tags = np.identity(5).astype(np.float32)
# 示例
exs = {path.stem: np.clip(np.load(path)[0], 0, ceils[path.stem[-1]]) for path in Path('examples').glob('*.npy')}
# 微调数据量
n_turning = 1000
# 多模块共享变量
class vars:
    WINDOW_SIZE = 1024
    WINDOW_STRIDE = 512

ukdale_channels = {
    'k': {1: [10], 2: [8], 5: [18]},
    'm': {1: [13], 2: [15], 5: [23]},
    'd': {1: [6], 2: [13], 5: [22]},
    'w': {1: [5], 2: [12], 5: [24]},
    'f': {1: [12], 2: [14], 5: [19]}
}

refit_channels = {
    'k':{2: [8], 5: [8], 6: [7]},
    'm':{2: [5], 5: [7], 6: [6]},
    'd':{2: [3], 5: [4], 6: [3]},
    'w':{2: [2], 5: [3], 6: [2]},
    'f':{2: [1], 5: [1], 6: [1]}
}

class ApplianceDataset(Dataset):
    def __init__(self, app_abb, mains, apps, noweight, stage) -> None:
        super().__init__()
        self.app_abb= app_abb
        self.mains = mains
        self.apps = apps
        self.weight = 1 if noweight else weight[ids[app_abb]]
        # 从效果上看，正负样本平衡反而会带来负收益
        # if stage == 'fit':
        #     # balance the number of samples
        #     pos_idx = np.nonzero(np.any(self.apps >= self.thresh, axis=1))[0]
        #     neg_idx = np.nonzero(np.any(self.apps < self.thresh, axis=1))[0]
        #     if 1 * len(pos_idx) < len(neg_idx):
        #         neg_idx = np.random.choice(neg_idx, 1 * len(pos_idx), replace=False)
        #         self.mains = np.concatenate([self.mains[pos_idx], self.mains[neg_idx]])
        #         self.apps = np.concatenate([self.apps[pos_idx], self.apps[neg_idx]])

    def __len__(self):
        return len(self.mains)
    
    def __getitem__(self, index):
        return ids[self.app_abb], self.mains[index], self.apps[index], tags[[ids[self.app_abb]]], self.weight, threshs[self.app_abb], ceils[self.app_abb]
    
def read(set_name, house_id, app_abb=None, channel=None):
    """ read a dataframe of a specific appliance or mains """
    if set_name == 'ukdale':
        if not app_abb: channel = 1
        path = DIR / set_name / f'house_{house_id}' / f'channel_{channel}.dat'
        df = pd.read_csv(path, sep=" ", header=None).iloc[:, :2]
    elif set_name == 'refit':
        path = DIR / set_name / f'CLEAN_House{house_id}.csv'
        df = pd.read_csv(path)
        column = 'Aggregate' if not app_abb else f'Appliance{channel}'
        df = df[['Unix', column]]
    df.columns = ["stamp", "power" if not app_abb else app_abb]
    df.sort_values(by="stamp")
    df['stamp'] = pd.to_datetime(df['stamp'], unit='s')
    df = df.set_index('stamp')
    df = df.resample('6s').mean().ffill(limit=30).dropna()
    df[df < (threshs[app_abb] if app_abb else 15)] = 0
    df = df.clip(lower=0, upper=6000 if not app_abb else ceils[app_abb])
    return df

def load_data(set_name, house_id):
    """ 加载指定房屋的总线曲线 & 5类电器支线曲线，并返回所有曲线时间戳有交集的部分 """
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / hashlib.sha256(f'{set_name}{house_id}'.encode()).hexdigest()
    if temp_path.exists():
        powers = np.load(temp_path)
    else:
        powers = read(set_name, house_id)
        for app_abb in 'kmdwf':
            channels = ukdale_channels[app_abb][house_id] if set_name == 'ukdale' else refit_channels[app_abb][house_id]
            apps = read(set_name, house_id, app_abb, channels[0])
            powers = pd.merge(powers, apps, on='stamp')
        powers = powers.to_numpy(dtype=np.float32)
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(temp_path, powers)
    return powers

def get_house_sets(set_name, house_id, noweight, stage):
    """ 获取单个房屋中各电器的数据集 """
    powers= load_data(set_name, house_id)
    if stage == 'fit' and set_name == 'ukdale' and house_id == 1: powers = powers[0: int(0.15 * len(powers))]
    powers = [np.copy(np.lib.stride_tricks.sliding_window_view(power, vars.WINDOW_SIZE)[::vars.WINDOW_STRIDE]).astype(np.float32) for power in powers.T]
    return [ApplianceDataset(app_abb, powers[0], powers[idx+1], noweight, stage) for idx, app_abb in enumerate("kmdwf")]

def get_houses_sets(set_houses, noweight, stage):
    """ 获取多个房屋中各电器的数据集 """
    datasets = [[], [], [], [], []]
    match = re.match(r'^(\D+)(\d+)$', set_houses)
    set_name, house_ids = match.groups()
    for house_id in house_ids:
        for idx, app_set in enumerate(get_house_sets(set_name, int(house_id), noweight, stage)):
            datasets[idx].append(app_set)
    datasets = list(map(lambda x: ConcatDataset(x), datasets))
    return datasets

def get_syn_house_sets(set_name, house_id, noweight):
    """" 生成一个房屋的合成数据 """
    def shift(power):
        # 9成概率offset在[-256, 256], 1成概率offset在[-512, -256]和[256, 512]
        offset = np.randint(-vars.WINDOW_SIZE//4, vars.WINDOW_SIZE//4) if np.random.random() < 0.8 else np.randint(-vars.WINDOW_SIZE//2, vars.WINDOW_SIZE//2)
        return np.concatenate([np.zeros(offset, dtype=np.float32), power[:-offset]] if offset > 0 else [power[-offset:], np.zeros(-offset, dtype=np.float32)])
        
    powers= load_data(set_name, house_id)
    powers = [np.copy(np.lib.stride_tricks.sliding_window_view(power, vars.WINDOW_SIZE)[::vars.WINDOW_STRIDE]).astype(np.float32) for power in powers.T]
    for idx, app_abb in enumerate("kmdwf"):
        # 只选取目标房屋没有目标电器的段
        mains = [agg for agg, app in zip(powers[0], powers[idx+1]) if np.max(app) < threshs[idx]]
        print(f"{set_name}-{house_id}-{app+abb}: {len(mains)}")
        # 随机选取指定数量的总线窗口
        random_ids = np.random.permutation(len(mains))
        mains = [mains[random_ids[i%random_ids]] for i in range(n_turning)]
        # 随机生成指定数量的支线窗口
        app_exs = exs[f'{set_name}{house_id}-{app_abb}']
        random_ids = np.random.permutation(len(app_exs))
        apps = [shift(app_exs[random_ids[i%random_ids]]) if i % 2 == 0 else np.zeros(vars.WINDOW_SIZE, dtype=np.float32) for i in range(n_turning)]
        mains = [agg + app for agg, app in zip(mains, apps)]
        yield ApplianceDataset(app_abb, np.array(mains), np.array(apps), noweight, 'turn')

def get_syn_houses_sets(set_houses, noweight)
    """ 获取合成的数据，用于微调 """
    datasets = [[], [], [], [], []]
    match = re.match(r'^(\D+)(\d+)$', set_houses)
    # 遍历每个房屋的每个设备
    for house_id in house_ids:
        for idx, app_set in enumerate(get_syn_house_sets(set_name, int(house_id), noweight)):
            datasets[idx].append(app_set)
    datasets = list(map(lambda x: ConcatDataset(x), datasets))
    return datasets