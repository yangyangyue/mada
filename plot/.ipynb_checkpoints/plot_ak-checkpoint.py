from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=8, suppress=True)

plt.rc("font", family="Times New Roman")

lines = []
# 初始化存储所有点的列表
for app_abb in "kmdwf":
    lines.append(np.load(Path('examples') / f'refit2-{app_abb}.npy')[0])
agg = np.stack(lines).sum(axis=0) + 10

fig= plt.figure(figsize=(16, 2), facecolor='none', edgecolor='none')
plt.plot(agg, linewidth=0.5, label='mains')
plt.plot(lines[0], linewidth=0.5, label='kettle')
plt.xlabel('Time(8s)')
plt.ylabel('Power(W)')
plt.legend()
# plt.show()
plt.savefig('ak_match.png',  transparent=True, bbox_inches='tight',  dpi=300)
