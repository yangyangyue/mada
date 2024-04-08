from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
x =np.loadtxt('results/')
plt.figure()
plt.plot(x[0, :])
plt.plot(x[1: ])
plt.plot(x[2:, ])
plt.show()


# from sconf import Config
# import numpy as np
# import matplotlib.pyplot as plt

# from dataset import NilmDataset

# config = Config('config.yaml')
# train_set = NilmDataset({ "ukdale": ["house_5"] }, ['kettle'], config.data_dir, config.app_alias, config.app_threshs)
# app_list = []
# for app_thresh, example, sample, app in train_set:
#     app_list.append(app)
# apps = np.array(app_list)
# print(np.mean(app_list))
#     # if np.max(app)< 60:
#     #     continue
#     # plt.figure()
#     # plt.plot(example, label='example')
#     # plt.plot(sample, label='sample')
#     # plt.plot(app, label='app')
#     # plt.legend()
#     # plt.show()
#     # plt.close()


# 问题：模型管理、模型特化、新设备必须重新训练设备模型
# 其实，也可以直接检索增强，不固定example，直接从知识库中搜索