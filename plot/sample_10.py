from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, random_split, Subset

from dataset import *

train_set, val_set = random_split(dataset, [0.8, 0.2])
# show
for i in range(10):
    app_id, agg, app, ex, weight, thresh, ceil = dataset[random.randint(0, len(dataset))]
    fig = plt.figure(figsize=(5, 3), facecolor='none', edgecolor='none')
    plt.plot(agg, label='mains')
    plt.plot(app, label=names[app_id])
    plt.xlabel('Time(6s)')
    plt.ylabel('Power(W)')
    plt.legend()
    plt.savefig(f'case/case{i}.png',  bbox_inches='tight',  dpi=300)
    plt.close()