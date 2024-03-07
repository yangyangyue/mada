"""load the power of main channel and the specific appliances. in general, only the `get_loaders()` is exported,
which returns the train loader and val loader in train stage but returns the only test loader in test stage. Besides,
mean/std of main channel and appliance channel is returned as well in train stage. The behavior of this module is
highly depends on the config file.

written by lily
email: lily231147@gmail.com
"""
import os
import typing
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from util.config import ReddConfig, UkdaleConfig


def get_loaders(app_name: str, config: typing.Union[ReddConfig, UkdaleConfig], train: bool = True):
    """ get data loaders and the mean/std of main channel and appliance channel """
    def segment(seq):
        """ segment the seq as slide window"""
        return np.lib.stride_tricks.sliding_window_view(seq, config.window_size)[::config.window_stride]

    threshold, min_on, min_off = config.threshold[app_name], config.min_on[app_name], config.min_off[app_name]
    house_indicies = config.train_houses if train else config.test_houses
    house_ratios = config.train_ratios if train else config.test_ratios

    # 1. get mains power, appliances power and appliances status of all candidate houses
    mains, apps, status = \
        np.empty((0, config.window_size)), np.empty((0, config.window_size)), np.empty((0, config.window_size))
    for house_id, house_ratio in zip(house_indicies, house_ratios):
        new_mains, new_apps = __load_data(app_name, house_id, config)
        new_status = __compute_status(new_apps, threshold, min_on, min_off)
        # segment the sequences using sliding window
        new_mains, new_apps, new_status = segment(new_mains), segment(new_apps), segment(new_status)
        # select the segments depends on the house_ratio with shuffling at train stage
        select_num = int(house_ratio * (num := len(new_mains)))
        ids = np.random.permutation(num)[:select_num] if train else np.arange(num)[:select_num]
        new_mains, new_apps, new_status = new_mains[ids], new_apps[ids], new_status[ids]
        mains, apps, status = \
            np.concatenate([mains, new_mains]), np.concatenate([apps, new_apps]), np.concatenate([status, new_status])
    # log some basic info of the data
    print(f"{app_name} load finished. num: {len(status)}, active num: {np.sum(np.max(status, axis=1) > 0)}")
    if train:
        # 2. get the mean/std of main channel and appliance channel, which is 0, 1, 0, 1 if not norm
        main_mean, main_std, app_mean, app_std = 0, 1, 0, 1
        if config.norm:
            main_mean, main_std, app_mean, app_std = np.mean(mains), np.std(mains), np.mean(apps), np.std(apps)
        # 3. split the train set and val set and then create the dataset, from which generate the data loader
        split_point = int((1 - config.val_size) * len(mains))
        train_set = NilmDataset(mains[:split_point], apps[:split_point], status[:split_point])
        val_set = NilmDataset(mains[split_point:], apps[split_point:], status[split_point:])
        train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size)
        val_loader = DataLoader(dataset=val_set, batch_size=config.batch_size)
        return train_loader, val_loader, (main_mean, main_std, app_mean, app_std)
    else:
        # 4. create dataset and then generate the data loader
        test_set = NilmDataset(mains, apps, status)
        test_loader = DataLoader(dataset=test_set, batch_size=config.batch_size)
        return test_loader


class NilmDataset(Dataset):
    """ dataset used to convert `np.ndarray` to `torch.FloatTensor`. the data generated is of shape `(N, 1, L)` """

    def __init__(self, mains: np.ndarray, apps: np.ndarray, status: np.ndarray):
        self.mains, self.apps, self.status = mains[:, None, :], apps[:, None, :], status[:, None, :]

    def __len__(self):
        return len(self.mains)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.mains[idx]), \
               torch.FloatTensor(self.apps[idx]), \
               torch.FloatTensor(self.status[idx])


def __compute_status(y, threshold, min_on, min_off):
    """ compute the on/off status of appliance based on its power sequence and threshold """
    status = np.zeros(y.shape)

    # 1. get the on/off status of each point
    initial_status = y >= threshold

    # 2. get the on/off event index
    status_diff = np.diff(initial_status, prepend=False, append=False)
    events_idx = status_diff.nonzero()[0].reshape((-1, 2))
    on_events, off_events = events_idx[:, 0], events_idx[:, 1]

    # 3. filter the on/off events to guarantee all on/off duration greater than its min duration thresh
    if len(on_events) == 0:
        return status
    off_duration = on_events[1:] - off_events[:-1]
    on_events = np.concatenate([on_events[0:1], on_events[1:][off_duration > min_off]])
    off_events = np.concatenate([off_events[:-1][off_duration > min_off], off_events[-1:]])
    on_duration = off_events - on_events
    on_events = on_events[on_duration >= min_on]
    off_events = off_events[on_duration >= min_on]

    # 4. reset the status off the app based on the filtered on_events and off_events
    for on, off in zip(on_events, off_events):
        status[on: off] = 1

    return status


def __read_power(path: str, sampling: str):
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


def __add_power(power1, power2):
    """ merge 2 DataFrames on their time index and then add their power """
    power = pd.merge(power1, power2, on='time')
    power.iloc[:, 0] = power.iloc[:, 0] + power.iloc[:, 1]
    return power.iloc[:, 0: 1]


def __load_data(app_name: str, house_id: int, config: typing.Union[ReddConfig, UkdaleConfig]):
    # get hyper parameters
    set_name = config.set_name
    base_dir = config.base_dir
    sampling = config.sampling
    cutoff = [config.cutoff[i] for i in ['aggregate', app_name]]

    # 1. read the main power of the house
    house_folder = f"{base_dir}house_{house_id}/"
    if set_name == "redd":
        main_1 = __read_power(f"{house_folder}channel_1.dat", sampling)
        main_2 = __read_power(f"{house_folder}channel_2.dat", sampling)
        house_data = __add_power(main_1, main_2)
    elif set_name == "ukdale":
        house_data = __read_power(f"{house_folder}channel_1.dat", sampling)
    else:
        raise ValueError("dataset is either redd or ukdale")

    # 2. get the channel of specified appliance
    house_apps = pd.read_csv(f"{house_folder}labels.dat", sep=' ', header=None).values
    app_channels = house_apps[(house_apps[:, 1] == app_name) |
                              (np.isin(house_apps[:, 1], config.app_alias.get(app_name))), 0]

    # 3. merge main power and appliance power
    app_data = house_data.copy()
    app_data.iloc[:, 0] = 0
    for channel in app_channels:
        new_data = __read_power(f"{house_folder}channel_{channel}.dat", sampling)
        app_data = __add_power(app_data, new_data)
    house_data = pd.merge(house_data, app_data, on='time')
    house_data.columns = ['aggregate', app_name]
    # 4. process data including dropna, clip, etc.
    house_data = house_data.dropna()
    house_data[house_data < 5] = 0
    house_data = house_data.clip(lower=0, upper=cutoff, axis=1).values
    return house_data[:, 0], house_data[:, 1]
