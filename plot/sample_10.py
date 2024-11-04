import sys
# sys.path.append('..')
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import ConcatDataset, random_split, Subset

from dataset import names, get_syn_houses_sets, get_houses_sets

# turn_datasets = get_houses_sets('ukdale2', False, 'fit')[-1]
turn_datasets = get_syn_houses_sets('ukdale2', False)[-1]
# show
for i in range(10):
    app_id, agg, app, ex, w,  _, _ = turn_datasets[np.random.randint(0, len(turn_datasets)-1)]
    fig = plt.figure(figsize=(5, 3), facecolor='none', edgecolor='none')
    plt.plot(agg, label='mains')
    plt.plot(app, label=names[app_id])
    # # offset = agg-app
    # # z = offset + np.array((app>50)*30)
    # # plt.plot(offset, label='xx')
    # # plt.plot(z, label='yy')
    plt.xlabel('Time(6s)')
    plt.ylabel('Power(W)')
    plt.legend()
    plt.savefig(f'../case/case{i}.png',  bbox_inches='tight',  dpi=300)
    # plt.close()