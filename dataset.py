"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import hashlib
from pathlib import Path
import random
import re
import tempfile

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset

DIR = Path('/hy-tmp/nilm_lf')
WINDOW_SIZE = 1024
WINDOW_STRIDE = 512

threshs ={"k": 2000, "m": 200, "d": 20, "w": 1200, "f": 50}
ceils = {"k": 3100, "m": 3000, "d": 2500, "w": 2500, "f": 300}

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

exs = [{}, {}, {}, {}, {}]
for path in Path('examples').glob('*.npy'):
    if   '-k' in path.stem: exs[0][path.stem] = np.clip(np.load(path)[0], 0, ceils['k'])
    elif '-m' in path.stem: exs[1][path.stem] = np.clip(np.load(path)[0], 0, ceils['m'])
    elif '-d' in path.stem: exs[2][path.stem] = np.clip(np.load(path)[0], 0, ceils['d'])
    elif '-w' in path.stem: exs[3][path.stem] = np.clip(np.load(path)[0], 0, ceils['w'])
    elif '-f' in path.stem: exs[4][path.stem] = np.clip(np.load(path)[0], 0, ceils['f'])
    else: pass

# refit_channels = {
#     'k':{2: [8], 3: [9], 4: [9], 5: [8], 6: [7], 7: [9], 8: [9], 9: [7], 11: [7], 12: [6], 13: [9], 17: [8], 19: [5], 20: [9], 21: [7]},
#     'm':{2: [5], 3: [8], 4: [8], 5: [7], 6: [6], 8: [8], 9: [6], 10: [8], 11: [6], 12: [5], 13:[7, 8], 15: [7], 17: [7], 18: [9], 19: [4], 20: [8]},
#     'd':{1: [6], 2: [3], 3: [5], 5: [4], 6: [3], 7: [6], 9: [4], 10:[6], 11: [4], 13: [4], 15: [4], 16: [6], 18: [6], 20: [5], 21: [4]},
#     'w':{1: [4, 5], 2: [2], 3: [6], 4: [4, 5], 5:[3], 6: [2], 7: [5], 8: [3, 4], 9: [2, 3], 10: [5], 11: [3], 13: [3], 15: [3], 16: [5], 
#          17: [4], 18:[4,5], 19:[2], 20: [4], 21: [3]},
#     'f':{1: [1, 2, 3], 2:[1], 3: [2, 3], 4:[1, 2, 3], 5: [1], 6: [1], 7:[1, 2, 3], 8: [1, 2], 9: [1], 10: [4], 11: [1, 2], 12:[1], 13: [2], 
#          15: [1], 16: [1, 2], 17: [1, 2], 18: [1, 2, 3], 19: [1], 20: [1, 2], 21: [1]}   
# }

class ApplianceDataset(Dataset):
    def __init__(self, app_abb, mains, apps, ex, stage) -> None:
        super().__init__()
        self.app_abb= app_abb
        self.mains = mains
        self.apps = apps
        self.ex = ex
        self.thresh = threshs[app_abb]
        self.ceil = ceils[app_abb]
        if stage == 'fit':
            # balance the number of samples
            pos_idx = np.nonzero(np.any(self.apps >= self.thresh, axis=1))[0]
            neg_idx = np.nonzero(np.any(self.apps < self.thresh, axis=1))[0]
            if 1 * len(pos_idx) < len(neg_idx):
                neg_idx = np.random.choice(neg_idx, 1 * len(pos_idx), replace=False)
                self.samples = np.concatenate([self.samples[pos_idx], self.samples[neg_idx]])
                self.apps = np.concatenate([self.apps[pos_idx], self.apps[neg_idx]])

    def __len__(self):
        return len(self.mains)
    
    def __getitem__(self, index):
        return self.mains[index], self.apps[index], self.ex, self.thresh, self.ceil
    
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
    df.columns = ["stamp", "power"]
    df.sort_values(by="stamp")
    df['stamp'] = pd.to_datetime(df['stamp'], unit='s')
    df = df.set_index('stamp')
    df = df.resample('6s').mean().ffill(limit=30).dropna()
    df[df < (threshs[app_abb] if app_abb else 15)] = 0
    df = df.clip(lower=0, upper=6000 if not app_abb else ceils[app_abb])
    return df

def load_data(set_name, house_id):
    """
    加载指定房屋的总线曲线 & 5类电器支线曲线，并返回所有曲线时间戳有交集的部分

    Args:
        set_name (str): 数据集名称，可选'ukdale'或'refit'
        house_id (int): 房屋ID

    Returns:
        np.ndarray: 加载的功率数据
    """
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

def get_house_sets(set_name, house_id, stage):
    powers= load_data(set_name, house_id)
    if stage == 'fit' and set_name == 'ukdale' and house_id == 1: powers = powers[0: int(0.15 * len(powers))]
    assert len(powers) >= WINDOW_SIZE
    mains, ks, ms, ds, ws, fs = powers[:, 0], powers[:, 1], powers[:, 2], powers[:, 3], powers[:, 4], powers[:, 5]
    mains = np.copy(np.lib.stride_tricks.sliding_window_view(mains, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32)
    ks = np.copy(np.lib.stride_tricks.sliding_window_view(ks, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32)
    ms = np.copy(np.lib.stride_tricks.sliding_window_view(ms, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32)
    ds = np.copy(np.lib.stride_tricks.sliding_window_view(ds, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32)
    ws = np.copy(np.lib.stride_tricks.sliding_window_view(ws, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32)
    fs = np.copy(np.lib.stride_tricks.sliding_window_view(fs, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32)
    if stage == "fit":
        k_set = ApplianceDataset('k', mains, ks, random.choice(list(exs[0].values())), stage)
        m_set = ApplianceDataset('m', mains, ms, random.choice(list(exs[1].values())), stage)
        d_set = ApplianceDataset('d', mains, ds, random.choice(list(exs[2].values())), stage)
        w_set = ApplianceDataset('w', mains, ws, random.choice(list(exs[3].values())), stage)
        f_set = ApplianceDataset('f', mains, fs, random.choice(list(exs[4].values())), stage)
    else:
        k_set = ApplianceDataset('k', mains, ks, exs[0][f"{set_name}{house_id}"], stage)
        m_set = ApplianceDataset('m', mains, ms, exs[1][f"{set_name}{house_id}"], stage)
        d_set = ApplianceDataset('d', mains, ds, exs[2][f"{set_name}{house_id}"], stage)
        w_set = ApplianceDataset('w', mains, ws, exs[3][f"{set_name}{house_id}"], stage)
        f_set = ApplianceDataset('f', mains, fs, exs[4][f"{set_name}{house_id}"], stage)
    return k_set, m_set, d_set, w_set, f_set


def get_houses_sets(set_houses, stage):
    """
    根据给定的房屋集合名称和阶段，获取多个房屋集合的数据集

    Args:
        set_houses (str): 房屋集合名称，如'ukdale15'
        stage (str): 阶段名称，用于区分训练（'fit'）和其他阶段

    Returns:
        List[ConcatDataset]: 包含5类电器的数据集的列表，每类电器的数据集是来自指定房屋的ConcatDataset

    """
    datasets = [[], [], [], [], []]
    match = re.match(r'^(\D+)(\d+)$', set_houses)
    set_name, house_ids = match.groups()
    for house_id in house_ids:
        for idx, app_set in enumerate(get_house_sets(set_name, int(house_id), stage)):
            datasets[idx].append(app_set)
    datasets = map(lambda x: ConcatDataset(x), datasets)
    return datasets
    