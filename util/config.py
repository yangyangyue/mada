"""The config of experiment: 1. The train, val and test dataset is split based on the house. 1) For ukdale by
default, the 0.15 data selected randomly of house1 and the all data of house5 is used to train the model, from which
the last 0.2 part of data is the val set and the rest is train set. Then, the all data of house2 is used as test set.
In this case, the val set is a subset of house5 actually. It should note that the model process data by sliding
window, meaning that the output will be reconstructed to reduce the overlops among the slide windows, so the sliding
windows of val set and train set must be continue if the window stride lower than window size. By default and
recommended, set `window_stride == window_size` to avoid the overlop and reconstruct.

written by lily
email: lily231147@gmail.com
"""
from typing import Literal


class BaseConfig:
    """ basic config """
    method: Literal["vae", "s2p"] = 'vae'
    n_epoch = 100
    batch_size = 128
    lr = 1e-3
    optimizer: Literal["adam", "adamw", "sgd"] = 'adam'
    optimizer_args = {
        'adam': {'betas': [0.9, 0.99]},
        'adamw': {'weight_decay': 0.},
        'sgd': {'momentum': None}}
    scheduler = {'step_size': 40, 'gamma': 0.1}


class ReddConfig(BaseConfig):
    """ specific config for redd """
    set_name = 'redd'
    app_names = ['microwave', 'dishwasher']
    train_houses, train_ratios = [1, 5], [0.2, 1]
    test_houses, test_ratios = [2], [1]
    val_size = 0.2
    sampling = '6s'
    window_size, window_stride = 1024, 64
    norm = False
    base_dir = './data/redd/'
    cutoff = {'aggregate': 6000, 'refrigerator': 400, 'washer_dryer': 3500, 'microwave': 1800, 'dishwasher': 1200}
    threshold = {'refrigerator': 50, 'washer_dryer': 20, 'microwave': 200, 'dishwasher': 10}
    min_on = {'refrigerator': 10, 'washer_dryer': 300, 'microwave': 2, 'dishwasher': 300}
    min_off = {'refrigerator': 2, 'washer_dryer': 26, 'microwave': 5, 'dishwasher': 300}


class UkdaleConfig(BaseConfig):
    """ specific config for ukdale """
    set_name = 'ukdale'
    app_names = ['kettle', 'microwave', 'dishwasher']
    app_alias = {
        'dishwasher': ['dish_washer']
    }
    train_houses, train_ratios = [1, 5], [0.15, 1]
    test_houses, test_ratios = [2], [1]
    val_size = 0.2
    sampling = '6s'
    window_size, window_stride = 1024, 64
    norm = False
    base_dir = './data/ukdale/'
    cutoff = {'aggregate': 6000, 'kettle': 3100, 'fridge': 300, 'washing_machine': 2500, 'microwave': 3000,
              'dishwasher': 2500}
    threshold = {'kettle': 2000, 'fridge': 50, 'washing_machine': 20, 'microwave': 200, 'dishwasher': 10}
    min_on = {'kettle': 2, 'fridge': 10, 'washing_machine': 300, 'microwave': 2, 'dishwasher': 300}
    min_off = {'kettle': 0, 'fridge': 2, 'washing_machine': 26, 'microwave': 5, 'dishwasher': 300}
