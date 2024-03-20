"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""
import hashlib
from pathlib import Path
import tempfile

import sys
sys.path.append('/home/aistudio/external-libraries')

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset

WINDOW_SIZE = 1024
WINDOW_STRIDE = 256


def read_power(path, sampling):
    """If cached, read dataframe from cache, or read dataframe from path and cache it otherwise"""
    temp_dir = Path(tempfile.gettempdir())
    father_path = temp_dir / hashlib.sha256(str(path).encode()).hexdigest()
    if father_path.exists():
        df = pd.read_feather(father_path)
    else:
        father_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(path, sep=" ", header=None).iloc[:, :2]
        df.columns = ["stamp", "power"]
        df.sort_values(by="stamp")
        df.to_feather(father_path)
    # 2. set time index
    df['stamp'] = pd.to_datetime(df['stamp'], unit='s')
    df = df.set_index('stamp')
    # 3. resample, fillna and return
    return df.resample(sampling).mean().ffill(limit=30)

def add_power(power1, power2):
    """ merge 2 DataFrames on their time index and then add their power """
    power = pd.merge(power1, power2, on='stamp')
    power.iloc[:, 0] = power.iloc[:, 0] + power.iloc[:, 1]
    return power.iloc[:, 0: 1]


class AbstractDataset(Dataset):
    """ designed for one appliance in one house """
    def __init__(self, dir, set_name, house, app_name, app_alias) -> None:
        super().__init__()
        print(f'{house}={app_name}')
        self.dir = dir
        self.set_name = set_name
        self.house = house
        self.app_name = app_name
        self.app_alias = app_alias
        self.samples, self.apps = self.load_data()
        if len(self.samples) < WINDOW_SIZE:
            self.samples, self.apps, self.examples = [], [], []
        else:
            self.samples = np.copy(np.lib.stride_tricks.sliding_window_view(self.samples, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32)
            self.apps = np.copy(np.lib.stride_tricks.sliding_window_view(self.apps, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32)
            self.examples = np.copy(self.load_examples(len(self.samples))).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """ the shape of each part is  (window_size, ) """
        return self.examples[idx], self.samples[idx], self.apps[idx]
    
    def load_examples(self, n_examples):
        """
        return examples in random order

        Args:
            n_examples: the examples will be duplicated to meet the num
        """
        path = Path('examples') / f'{self.set_name}_{self.house}_{self.app_name}.csv'
        examples = np.loadtxt(path, dtype=np.float32)
        return np.tile(examples[None, :], (3, 1))
    
    def load_data(self, cutoff=6000, sampling='6s'):
        """ return samples and apps, this method is implemented by specific dataset"""
        raise NotImplementedError('this method must be implemented by sub-class!')
    
class UkdaleDataset(AbstractDataset):
    def __init__(self, dir, house, app_name, app_alias) -> None:
        super().__init__(dir, 'ukdale', house, app_name, app_alias)

    def load_data(self, cutoff=6000, sampling='6s'):
        dir = self.dir / self.house

        # 1. read the main power of the house
        house_data = read_power(dir / 'channel_1.dat', sampling)

        # 2. get the channel of specified appliance
        house_apps = pd.read_csv(dir / 'labels.dat', sep=' ', header=None).values
        app_channels = house_apps[np.isin(house_apps[:, 1], self.app_alias), 0]
        if len(app_channels) == 0:
            return [], []

        # 3. merge main power and appliance power
        app_data = house_data.copy()
        app_data.iloc[:, 0] = 0
        for channel in app_channels:
            new_data = read_power(dir / f"channel_{channel}.dat", sampling)
            app_data = add_power(app_data, new_data)
        house_data = pd.merge(house_data, app_data, on='stamp')
        house_data.columns = ['aggregate', self.app_name]
        # 4. process data including dropna, clip, etc.
        house_data = house_data.dropna()
        house_data[house_data < 5] = 0
        house_data = house_data.clip(lower=0, upper=cutoff, axis=1).to_numpy(dtype=np.float32)
        return house_data[:, 0], house_data[:, 1]
    
class ReddDataset(AbstractDataset):
    def __init__(self, dir, house, app_name, app_alias) -> None:
        super().__init__(dir, 'redd', house, app_name, app_alias)

    def load_data(self, cutoff=6000, sampling='6s'):
        dir = self.dir / self.house

        # 1. read the main power of the house
        main_1 = read_power(dir / 'channel_1.dat', sampling)
        main_2 = read_power(dir / 'channel_2.dat', sampling)
        house_data = add_power(main_1, main_2)

        # 2. get the channel of specified appliance
        house_apps = pd.read_csv(dir / f"labels.dat", sep=' ', header=None).values
        app_channels = house_apps[np.isin(house_apps[:, 1], self.app_alias), 0]
        if len(app_channels) == 0:
            return [], []

        # 3. merge main power and appliance power
        app_data = house_data.copy()
        app_data.iloc[:, 0] = 0
        for channel in app_channels:
            new_data = read_power(dir / f'channel_{channel}.dat', sampling)
            app_data = add_power(app_data, new_data)
        house_data = pd.merge(house_data, app_data, on='stamp')
        house_data.columns = ['aggregate', self.app_name]
        # 4. process data including dropna, clip, etc.
        house_data = house_data.dropna()
        house_data[house_data < 5] = 0
        house_data = house_data.clip(lower=0, upper=cutoff, axis=1).to_numpy(dtype=np.float32)
        return house_data[:, 0], house_data[:, 1]


class NilmDataset(Dataset):
    """
    build unified dataset for all appliances
    """
    def __init__(self, config, train=True):
        if train:
            houses = config.train_houses
            apps = config.train_apps
        else:
            houses =  config.test_houses
            apps = config.test_apps

        datasets = []
        # build dataset for appliances in ukdale
        if 'ukdale' in houses:
            dir = Path(config.data_dir) / 'ukdale'
            app_alias = config.app_alias['ukdale']
            datasets += [UkdaleDataset(dir,  house, app_name, app_alias[app_name]) 
                            for house in houses['ukdale'] for app_name in apps['ukdale']]
        # build dataset for appliances in redd
        if 'redd' in houses:
            dir = Path(config.data_dir) / 'redd'
            app_alias = config.app_alias['redd']
            datasets += [ReddDataset(dir, house, app_name, app_alias[app_name]) 
                            for house in houses['redd'] for app_name in apps['redd']]
        self.dataset = ConcatDataset(datasets)
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    