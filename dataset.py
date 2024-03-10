"""load the power of main channel and the specific appliances. in general, only the `get_loaders()` is exported,
which returns the train loader and val loader in train stage but returns the only test loader in test stage. Besides,
mean/std of main channel and appliance channel is returned as well in train stage. The behavior of this module is
highly depends on the config file.

written by lily
email: lily231147@gmail.com
"""
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset


class AbstractDataset(Dataset):
    """ designed for one appliance in one house """
    def __init__(self, app_name, set_name, house_id, window_size, window_stride) -> None:
        super().__init__()
        self.set_name = set_name
        self.app_name = app_name
        self.window_size = window_size
        self.window_stride = window_stride
        self.samples, self.apps = self.load_data(app_name, set_name, house_id)
        
        self.samples = np.lib.stride_tricks.sliding_window_view(self.samples, window_size)[::window_stride]
        self.apps = np.lib.stride_tricks.sliding_window_view(self.apps, window_size)[::window_stride]
        self.examples = self.load_examples(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.examples[idx], self.samples[idx], self.apps[idx]
    
    def load_examples(self, n_examples):
        files = glob.glob(f'examples/{self.set_name}_{self.app_name}_*')
        examples = [np.loadtxt(file) for file in files]
        examples = examples * (n_examples // len(examples)) + examples[: n_examples % len(examples)]
        return np.random.shuffle(np.array(examples))
    
    def load_data(self, app_name, house_id, app_alias, cutoff, sampling):
        ...

    def read_power(path: str, sampling: str):
        """ read the data of path to DataFrame, set its time index, and resample it"""
        # 1. just import the data from disk to memory. the data file is preferred of type `.father` for faster reading
        if os.path.exists(father_path := path.replace('.dat', '.feather')):
            data = pd.read_feather(father_path)
        else:
            data = pd.read_csv(path, sep=' ', header=None)
            data.columns = ['time', 'power']
            data.to_feather(father_path)
        # 2. set time index
        data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], unit='s')
        data = data.set_index('time')
        # 3. resample, fillna and return
        return data.resample(sampling).mean().fillna(method='ffill', limit=30)

    def add_power(power1, power2):
        """ merge 2 DataFrames on their time index and then add their power """
        power = pd.merge(power1, power2, on='time')
        power.iloc[:, 0] = power.iloc[:, 0] + power.iloc[:, 1]
        return power.iloc[:, 0: 1]
    
class UkdaleDataset(AbstractDataset):
    def __init__(self, app_name, set_name, house_id, window_size, window_stride) -> None:
        super().__init__(app_name, set_name, house_id, window_size, window_stride)

    def load_ukdale(self, app_name: str, house_id: int, app_alias, cutoff, sampling):
        base_dir = './data/ukdale/'

        # 1. read the main power of the house
        house_folder = f"{base_dir}house_{house_id}/"
        house_data = self.read_power(f"{house_folder}channel_1.dat", sampling)

        # 2. get the channel of specified appliance
        house_apps = pd.read_csv(f"{house_folder}labels.dat", sep=' ', header=None).values
        app_channels = house_apps[(house_apps[:, 1] == app_name) |
                                (np.isin(house_apps[:, 1], app_alias.get(app_name))), 0]

        # 3. merge main power and appliance power
        app_data = house_data.copy()
        app_data.iloc[:, 0] = 0
        for channel in app_channels:
            new_data = self.read_power(f"{house_folder}channel_{channel}.dat", sampling)
            app_data = self.add_power(app_data, new_data)
        house_data = pd.merge(house_data, app_data, on='time')
        house_data.columns = ['aggregate', app_name]
        # 4. process data including dropna, clip, etc.
        house_data = house_data.dropna()
        house_data[house_data < 5] = 0
        house_data = house_data.clip(lower=0, upper=cutoff, axis=1).values
        return house_data[:, 0], house_data[:, 1]
    
class ReddDataset(AbstractDataset):
    def __init__(self, app_name, set_name, house_id, window_size, window_stride) -> None:
        super().__init__(app_name, set_name, house_id, window_size, window_stride)

    def load_redd(self, app_name: str, house_id: int, app_alias, cutoff, sampling):
        base_dir = './data/redd/'

        # 1. read the main power of the house
        house_folder = f"{base_dir}house_{house_id}/"
        main_1 = self.read_power(f"{house_folder}channel_1.dat", sampling)
        main_2 = self.read_power(f"{house_folder}channel_2.dat", sampling)
        house_data = self.add_power(main_1, main_2)

        # 2. get the channel of specified appliance
        house_apps = pd.read_csv(f"{house_folder}labels.dat", sep=' ', header=None).values
        app_channels = house_apps[(house_apps[:, 1] == app_name) |
                                (np.isin(house_apps[:, 1], app_alias.get(app_name))), 0]

        # 3. merge main power and appliance power
        app_data = house_data.copy()
        app_data.iloc[:, 0] = 0
        for channel in app_channels:
            new_data = self.read_power(f"{house_folder}channel_{channel}.dat", sampling)
            app_data = self.add_power(app_data, new_data)
        house_data = pd.merge(house_data, app_data, on='time')
        house_data.columns = ['aggregate', app_name]
        # 4. process data including dropna, clip, etc.
        house_data = house_data.dropna()
        house_data[house_data < 5] = 0
        house_data = house_data.clip(lower=0, upper=cutoff, axis=1).values
        return house_data[:, 0], house_data[:, 1]


class NilmDataset(Dataset):
    """ dataset used to convert `np.ndarray` to `torch.FloatTensor`. the data generated is of shape `(N, 1, L)` """

    def __init__(self, config):
        self.mains, self.apps, self.status = mains[:, None, :], apps[:, None, :], status[:, None, :]

    def __len__(self):
        return len(self.mains)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.mains[idx]), \
               torch.FloatTensor(self.apps[idx]), \
               torch.FloatTensor(self.status[idx])





