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

WINDOW_SIZE = 1024
WINDOW_STRIDE = 256

balance = False

threshs ={"k": 2000, "m": 200, "d": 10, "w": 20, "f": 50}
alias = {
    "k": ["kettle"],
    "m": ["microwave"],
    "d": ["dishwasher", "dish_washer", "dishwaser"],
    "w": ["washing_machine", "washer_dryer"],
    "f": ["fridge", "fridge_freezer", "refrigerator"],
}



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
    def __init__(self, dir, set_name, house, app_abb) -> None:
        super().__init__()
        self.dir = dir
        self.set_name = set_name
        self.house = house
        self.app_alias = alias[app_abb]
        self.app_thresh = threshs[app_abb]

        print(f'{set_name}-{house}-{app_abb} loading...')
        self.samples, self.apps = self.load_data()
        if len(self.samples) < WINDOW_SIZE:
            self.samples, self.apps, self.example = [], [], None
        else:
            self.samples = np.copy(np.lib.stride_tricks.sliding_window_view(self.samples, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32)
            self.apps = np.copy(np.lib.stride_tricks.sliding_window_view(self.apps, WINDOW_SIZE)[::WINDOW_STRIDE]).astype(np.float32)
            path = Path('examples') / f'{self.set_name}{self.house[-1]}{app_abb}.csv'
            self.example = np.loadtxt(path, dtype=np.float32)

        # for ukdale house_1, only select 15% of data
        if balance and self.set_name == 'ukdale' and self.house == 'house_1':
            num = len(self.samples)
            ind = np.random.permutation(num)
            select_ids = ind[: int(0.15 * num)]
            self.samples = self.samples[select_ids]
            self.apps = self.apps[select_ids]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """ the shape of each part is  (window_size, ) """
        return self.app_thresh, self.example, self.samples[idx], self.apps[idx]
    
    def load_data(self, cutoff=6000, sampling='6s'):
        """ return samples and apps, this method is implemented by specific dataset """
        raise NotImplementedError('this method must be implemented by sub-class!')
    
class UkdaleDataset(AbstractDataset):
    def __init__(self, dir, house, app_abb) -> None:
        super().__init__(dir, 'ukdale', house, app_abb)

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
        # 4. process data including dropna, clip, etc.
        house_data = house_data.dropna()
        house_data[house_data < 5] = 0
        house_data = house_data.clip(lower=0, upper=cutoff, axis=1).to_numpy(dtype=np.float32)
        return house_data[:, 0], house_data[:, 1]
    
class ReddDataset(AbstractDataset):
    def __init__(self, dir, house, app_abb) -> None:
        super().__init__(dir, 'redd', house, app_abb)

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
        # 4. process data including dropna, clip, etc.
        house_data = house_data.dropna()
        house_data[house_data < 5] = 0
        house_data = house_data.clip(lower=0, upper=cutoff, axis=1).to_numpy(dtype=np.float32)
        return house_data[:, 0], house_data[:, 1]