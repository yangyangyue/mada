# Copyright 2023 lily. All rights reserved.
#
# Author: lily
# Email: lily231147@proton.me

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

import pandas as pd
from sconf import Config

from dataset import add_power, read_power


np.random.seed(42)
plt.rc("font", family="Times New Roman")

WINDOW_LENGTH = 4096
WINDOW_STRIDE = 1024

def load_data(set_name, house, app_alias, cutoff=6000, sampling='6s'):
    dir = Path('data') / set_name / house
    # 2. get the channel of specified appliance
    house_apps = pd.read_csv(dir / f"labels.dat", sep=' ', header=None).values
    app_channels = house_apps[np.isin(house_apps[:, 1], app_alias), 0]
    if len(app_channels) == 0:
        return [], []
    # 3. merge main power and appliance power
    app_data = read_power(dir / f'channel_{app_channels[0]}.dat', sampling)
    for channel in app_channels[1:]:
        new_data = read_power(dir / f'channel_{channel}.dat', sampling)
        app_data = add_power(app_data, new_data)
    # 4. process data including dropna, clip, etc.
    app_data = app_data.dropna()
    app_data[app_data < 5] = 0
    app_data = app_data.clip(lower=0, upper=cutoff, axis=1).values
    return app_data[:, 0]



def get_example(set_name, house, app_name, app_alias, thresh=30):
    """ save examples """
    def next(_):
        examples.append(apps_in_window)
        np.savetxt(f'examples/{set_name}_{house}_{app_name}.csv', np.array(examples).transpose(), fmt="%.2f")
        print(f'saved example {len(examples)}')

    examples = []
    apps = load_data(set_name, house, app_alias)
    for apps_in_window in sliding_window_view(apps, WINDOW_LENGTH)[::WINDOW_LENGTH]:
        if any(apps_in_window > thresh):
            fig = plt.figure(figsize=(4, 2.5), dpi=300)
            plt.ylabel("Power(W)")
            plt.xlabel("Time(6s)")
            # plot power curve
            plt.plot(apps_in_window, label=app_name)
            plt.legend(loc="upper right")
            # if click the figure, saving it
            fig.canvas.mpl_connect('button_press_event', next)
            plt.show()
            plt.close(fig)

if __name__ == '__main__':
    # 'ukdale', 'redd', 'refit'
    set_name = 'ukdale'
    # 'house_1', 'house_2', 'house_3' ...
    house = 'house_5'
    # 'kettle', 'microwave', 'dishwasher', 'washing_machine', 'fridge'
    app_name = 'fridge'

    config = Config('config.yaml')
    get_example(set_name, house, app_name, config.app_alias[set_name][app_name])
