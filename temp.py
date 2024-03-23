
from sconf import Config
import numpy as np
import matplotlib.pyplot as plt

from dataset import NilmDataset

config = Config('config.yaml')
train_set = NilmDataset({ "ukdale": ["house_5"] }, ['kettle'], config.data_dir, config.app_alias, config.app_threshs)
app_list = []
for app_thresh, example, sample, app in train_set:
    app_list.append(app)
apps = np.array(app_list)
print(np.mean(app_list))
    # if np.max(app)< 60:
    #     continue
    # plt.figure()
    # plt.plot(example, label='example')
    # plt.plot(sample, label='sample')
    # plt.plot(app, label='app')
    # plt.legend()
    # plt.show()
    # plt.close()