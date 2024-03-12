"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""
import glob
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset


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
    def __init__(self, set_name, house, app_name, app_alias, window_size, window_stride) -> None:
        super().__init__()
        self.set_name = set_name
        self.house = house
        self.app_name = app_name
        self.app_alias = app_alias
        self.samples, self.apps = self.load_data(house, app_name, app_alias)
        
        self.samples = np.lib.stride_tricks.sliding_window_view(self.samples, window_size)[::window_stride]
        self.apps = np.lib.stride_tricks.sliding_window_view(self.apps, window_size)[::window_stride]
        self.examples = self.load_examples(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.examples[idx], self.samples[idx], self.apps[idx]
    
    def load_examples(self, n_examples):
        """
        return examples in random order

        Args:
            n_examples: the examples form files will be duplicated to meet the num
        """
        files = glob.glob(f'examples/{self.set_name}_{self.house}_{self.app_name}_*')
        examples = [np.loadtxt(file, dtype=np.float32) for file in files]
        examples = examples * (n_examples // len(examples)) + examples[: n_examples % len(examples)]
        return np.random.shuffle(np.array(examples))
    
    def load_data(self, cutoff=6000, sampling='6s'):
        """ return samples and apps, this method is implemented by specific dataset"""
        raise NotImplementedError('this method must be implemented in sub-class!')
    
class UkdaleDataset(AbstractDataset):
    def __init__(self, house, app_name, app_alias, window_size, window_stride) -> None:
        super().__init__(house, app_name, app_alias, window_size, window_stride)

    def load_data(self, cutoff=6000, sampling='6s'):
        dir = Path('data') / 'ukdale' / self.house

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
        house_data = pd.merge(house_data, app_data, on='time')
        house_data.columns = ['aggregate', self.app_name]
        # 4. process data including dropna, clip, etc.
        house_data = house_data.dropna()
        house_data[house_data < 5] = 0
        house_data = house_data.clip(lower=0, upper=cutoff, axis=1).values
        return house_data[:, 0], house_data[:, 1]
    
class ReddDataset(AbstractDataset):
    def __init__(self, house, app_name, app_alias, window_size, window_stride) -> None:
        super().__init__(house, app_name, app_alias, window_size, window_stride)

    def load_data(self, cutoff=6000, sampling='6s'):
        dir = Path('data') / 'redd' / self.house

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
        house_data = pd.merge(house_data, app_data, on='time')
        house_data.columns = ['aggregate', self.app_name]
        # 4. process data including dropna, clip, etc.
        house_data = house_data.dropna()
        house_data[house_data < 5] = 0
        house_data = house_data.clip(lower=0, upper=cutoff, axis=1).values
        return house_data[:, 0], house_data[:, 1]


class NilmDataset(Dataset):
    """
    build unified dataset for specific appliances
    """
    def __init__(self, config, train=True):
        if train:
            houses = config.train_houses
            apps = config.train_apps
        else:
            houses =  config.test_houses
            apps = config.test_apps

        window_size, window_stride = config.window_size, config.window_stride

        datasets = []
        if 'ukdale' in houses:
            datasets += [UkdaleDataset(house, app_name, app_alias, window_size, window_stride) 
                            for house in houses['ukdale'] for app_name, app_alias in apps['ukdale'].items()]
        if 'redd' in houses:
            datasets += [ReddDataset(house, app_name, app_alias, window_size, window_stride) 
                            for house in houses['redd'] for app_name, app_alias in apps['redd'].items()]
        self.dataset = ConcatDataset(datasets)
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    