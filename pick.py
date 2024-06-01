"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dataset


np.random.seed(42)
plt.rc("font", family="Times New Roman")

WINDOW_LENGTH = 1024
WINDOW_STRIDE = 256
N = 1
dir = Path("C:\\Users\\21975\Downloads\\nilm_lf")

def get_example(data_dir, set_name, house_id, app_abb, channel, n):
    """ save examples """
    print(f"Need to get {n} examples for {set_name}{house_id}-{app_abb}{channel}")
    examples = []
    apps = dataset.read(data_dir, set_name, house_id, app_abb, channel).to_numpy(dtype=np.float32)[:, 0]
    def on_click(event):
        if not event.dblclick:
            return
        center = int(event.xdata)
        left = center - WINDOW_LENGTH // 2
        right = center + WINDOW_LENGTH // 2
        ax.set_xlim(left, right)
        fig.canvas.draw()
        examples.append(apps[left:right])
        print(f"Selected {len(examples)} examples")
        if len(examples) == n:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
    fig = plt.figure(figsize=(5, 3), dpi=300)
    ax = plt.gca()
    plt.ylabel("Power(W)")
    plt.xlabel("Time(6s)")
    plt.plot(apps, label=app_abb)
    plt.legend(loc="upper right")
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    assert len(examples) == n
    return examples

# get_example(dir, 'ukdale', 1, 'w', 5, N)
# get_example(dir, 'ukdale', 2, 'w', 12, N)
# get_example(dir, 'ukdale', 5, 'w', 24, N)
# get_example(dir, 'refit', 2, 'w', 2, N)
# get_example(dir, 'refit', 5, 'w', 3, N)
# get_example(dir, 'refit', 6, 'w', 2, N)

# for app_abb, house_channels in dataset.ukdale_channels.items():
#     for house_id, channels in house_channels.items():
#         if house_id == 2:
#             apps = dataset.read(dir, 'ukdale', house_id, app_abb, channels[0]).to_numpy(dtype=np.float32)[:, 0]
#             print(app_abb, np.mean(apps[apps < dataset.threshs[app_abb]]))


# if __name__ == '__main__':
#     for app_abb, house_channels in dataset.ukdale_channels.items():
#         for house_id, channels in house_channels.items():
#             examples = []
#             for i, channel in enumerate(channels):
#                 examples.extend(get_example(dir, 'ukdale', house_id, app_abb, channel, (N+i) // len(channels)))
#             np.save(Path('examples') / f'ukdale{house_id}-{app_abb}.npy', np.stack(examples))
#     for app_abb, house_channels in dataset.refit_channels.items():
#         for house_id, channels in house_channels.items():
#             examples = []
#             for i, channel in enumerate(channels):
#                 examples.extend(get_example(dir, 'refit', house_id, app_abb, channel, (N+i) // len(channels)))
#             np.save(Path('examples') / f'refit{house_id}-{app_abb}.npy', np.stack(examples))