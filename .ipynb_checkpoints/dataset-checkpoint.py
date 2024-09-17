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
WINDOW_SIZE = 1024
WINDOW_STRIDE = 512

ids = {"k": 0, "m": 1, "d": 2, "w": 3, "f": 4}
threshs = {"k": 2000, "m": 200, "d": 20, "w": 1200, "f": 50}
ceils = {"k": 3100, "m": 3000, "d": 2500, "w": 2500, "f": 300}
names = ['kettle', 'microwave', 'dishwasher', 'washing_machine', 'fridge']

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
  
def extract_features(signal, fs=1/6):  
    N = len(signal)  
    # 1. 统计特征  
    mean_val = np.mean(signal)
    std_dev = np.std(signal)  
    max_val = np.max(signal)
    # 2. 峰度：衡量信号分布的尖锐程度或平坦程度。
    moment2 = np.mean((signal - mean_val) ** 2)  
    moment3 = np.mean((signal - mean_val) ** 3)  
    moment4 = np.mean((signal - mean_val) ** 4)  
    kurtosis = moment4 / (moment2 ** 2) - 3
    # 3. 能量  
    energy = np.sum(signal ** 2)  
    # 4. 波形因子：信号的均方根与平均绝对值的比值，反映波形尖锐程度
    mean_abs = np.mean(np.abs(signal))  
    form_factor = np.sqrt(moment2) / mean_abs  
    # 5. 频域特征  
    fft_result = np.fft.fft(signal)  
    fft_magnitude = np.abs(fft_result)  
    fft_freq = np.fft.fftfreq(N, d=1/fs)  # 假设采样频率为fs，否则默认为1  
    # 频谱质心：所有频率分量的加权平均频率，反映了频谱的中心位置
    spectral_centroid = np.sum(fft_freq[fft_freq > 0] * fft_magnitude[fft_freq > 0]**2) / np.sum(fft_magnitude[fft_freq > 0]**2)  
    # 频谱带宽：频谱带宽：频谱能量分布的范围，这里简单使用标准差作为估计
    spectral_bandwidth = np.std(fft_freq[fft_freq > 0] * fft_magnitude[fft_freq > 0]**2) / np.mean(fft_magnitude[fft_freq > 0]**2)  
    return [mean_val, std_dev, max_val, kurtosis, energy, form_factor, spectral_centroid, spectral_bandwidth]

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
    def __init__(self, app_abb, mains, apps, ex, weight, stage) -> None:
        super().__init__()
        self.app_abb= app_abb
        self.mains = mains
        self.apps = apps
        self.ex = ex
        self.weight = weight
        self.thresh = threshs[app_abb]
        self.ceil = ceils[app_abb]
        # 从效果评估上来说，正负样本平衡反而会带来负收益
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
        return ids[self.app_abb], self.mains[index], self.apps[index], self.ex, self.weight, threshs[self.app_abb], ceils[self.app_abb]
    
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

def get_house_sets(set_name, house_id, exs, weights, stage):
    powers= load_data(set_name, house_id)
    if stage == 'fit' and set_name == 'ukdale' and house_id == 1: powers = powers[0: int(0.15 * len(powers))]
    powers = [np.copy(np.lib.stride_tricks.sliding_window_view(power, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32) for power in powers.T]
    return [ApplianceDataset(app_abb, powers[0], powers[idx+1], exs[idx], weights[idx], stage) for idx, app_abb in enumerate("kmdwf")]


def get_houses_sets(set_houses, onehot, noweight, stage):
    """
    根据给定的房屋集合名称和阶段，获取多个房屋集合的数据集

    Args:
        set_houses (str): 房屋集合名称，如'ukdale15'
        stage (str): 阶段名称，用于区分训练（'fit'）和其他阶段

    Returns:
        List[ConcatDataset]: 包含5类电器的数据集的列表，每类电器的数据集是来自指定房屋的ConcatDataset

    """
    # 计算各电器的损失权重：[1.00, 1.03, 1.24, 1.24, 10.33]
    weights = 1 / np.array(list(ceils.values()))
    weights = [1] * 5 if noweight else (weights / np.min(weights))
    # 获取负荷印记
    signs = {}
    max_values = [float('-inf')] * 8
    min_values = [float('inf')] * 8
    cs = {'k': 'blue', 'm': 'orange', 'd': 'green', 'w': 'red', 'f': 'purple'}
    for path in Path('examples').glob('*.npy'):
        signs[path.stem] = extract_features(np.clip(np.load(path)[0], 0, ceils[path.stem[-1]]))
        for i in range(8): max_values[i], min_values[i] = max(signs[path.stem][i], max_values[i]), min(signs[path.stem][i], min_values[i])
    for k, v in signs.items():
        for i in range(8): signs[k][i] = (signs[k][i] - min_values[i]) / (max_values[i] - min_values[i])
    tsne = TSNE(n_components=2, random_state=0, perplexity=5)  # 降到2维  
    X_tsne = tsne.fit_transform(np.array(list(signs.values())))
    plt.figure(figsize=(8, 6))  
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=[cs[stem[-1]] for stem in signs.keys()])  
    plt.title('t-SNE visualization of 20 feature vectors')  
    plt.xlabel('t-SNE feature 1')  
    plt.ylabel('t-SNE feature 2')  
    plt.grid(True)
    plt.savefig(f'case/xxx.png',  bbox_inches='tight',  dpi=300)
    exs = np.identity(5).astype(np.float32) if onehot else [np.concatenate([v for k,v in signs.items() if k[-1]==app_abb]).astype(np.float32) for app_abb in "kmdwf"]
    datasets = [[], [], [], [], []]
    match = re.match(r'^(\D+)(\d+)$', set_houses)
    set_name, house_ids = match.groups()
    for house_id in house_ids:
        for idx, app_set in enumerate(get_house_sets(set_name, int(house_id), exs, weights, stage)):
            datasets[idx].append(app_set)
    datasets = list(map(lambda x: ConcatDataset(x), datasets))
    return datasets
    