"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""
import sys
sys.path.append("..")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import *

np.random.seed(42)
# plt.rc("font", family="Times New Roman")

set_name = 'ukdale'
house_id = 2

powers = load_data(set_name, house_id).T

if set_name == 'ukdale' and house_id == 1: powers = powers[0: int(0.15 * len(powers))]
# powers = [np.copy(np.lib.stride_tricks.sliding_window_view(power, vars.WINDOW_SIZE)[::vars.WINDOW_STRIDE]).astype(np.float32) for power in powers.T]

plt.figure()
plt.plot(powers[0], label='mains')
plt.plot(powers[1], label='k')
plt.plot(powers[2], label='m')
plt.plot(powers[3], label='d')
plt.plot(powers[4], label='w')
plt.plot(powers[5], label='f')
plt.legend(loc='upper right')
plt.show()
# plt.savefig(f'../case/{set_name}_{house_id}.png', bbox_inches='tight',  dpi=300)
plt.close()