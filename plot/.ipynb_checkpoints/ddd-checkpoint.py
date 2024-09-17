from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=8, suppress=True)

plt.rc("font", family="Times New Roman")


mains = pd.read_csv(Path('C:\\Users\\21975\\Downloads\\nilm_lf') / 'ukdale' / 'house_2' /f'mains.dat', sep=" ", header=None).iloc[:, :2].to_numpy(dtype=np.float64)
# apps = pd.read_csv(Path('C:\\Users\\21975\\Downloads\\nilm_lf') / 'ukdale' / 'house_2' /f'channel_8.dat', sep=" ", header=None).iloc[:, :2].to_numpy(dtype=np.float64)


# k = np.load(Path('examples') / f'refit2-k.npy')[0]
# plt.plot(k, label='appliance')
# plt.plot([1800] * len(k), label='threshold')





# ..
# # k第一个事件位置
# idx = np.where(mains[:, 0] > 1366185684.0)[0][0]
# mains = mains[idx: idx+120, 1]
# plt.figure()
# plt.plot(mains, label='mains')
# plt.axvline(x=10, color='red', linestyle='--', label='event1')
# plt.axvline(x=88, color='red', linestyle='--', label='event2')
# # plt.axvline(x=33, color='green', label='event1')
# # plt.axvline(x=95, color='green', label='event2')
# plt.xlabel('Time(s)')
# plt.ylabel('Power(W)')
# plt.legend()
# plt.show()
# plt.savefig('k_main_accurate.png', bbox_inches='tight',  dpi=300)

def axvchar(x, ax, char, color, y_span=80):
    y_min, y_max = ax.get_ylim()  # 获取y轴的范围
    y_min += 0.02 * (y_max - y_min)  # 设定字符在y轴上的位置
    y_max -= 0.02 * (y_max - y_min)  # 设定字符在y轴上的位置

    while(y_min < y_max):
        ax.text(x, y_min, char, ha='center', va='center', color=color, fontsize=8, fontweight='bold')
        y_min += y_span  # 设置字符间隔

# ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']。
# k第一个事件位置
idx = np.where(mains[:, 0] > 1366185684.0)[0][0]
mains = mains[idx+50: idx+130, 1]
fig, ax = plt.subplots()
plt.plot(mains, label='mains')
# plt.axvline(x=10, color='red', linestyle='--', label='event1')
plt.axvline(x=33, color='red', linestyle='--', label='pre-timestamp')
axvchar(3, ax, '>', '#9467bd')
axvchar(63, ax, '<', '#9467bd')
colors = ['#8c564b', '#339966']
# 绘制方框
size = 2
amps = mains[size:] - mains[:-size]
i = 0
for idx, amp in enumerate(amps):
    if idx < 3 or idx >61:
        continue
    # if idx % 5 != 0:
    #     continue
    if idx == 44:
        rect = plt.Rectangle(xy=(idx, mains[idx]-(20 if amp > 0 else (-20))), width=2, height=amp+(40 if amp > 0 else (-40)), facecolor='none', edgecolor='#FF7C80', linewidth=1.2)
    else:
        if idx < 42 or idx > 46:
            if idx % 3 != 0:
                continue
        rect = plt.Rectangle(xy=(idx, mains[idx]-(20 if amp > 0 else (-20))), width=2, height=amp+(40 if amp > 0 else (-40)), facecolor='none', edgecolor=colors[i % 2], linewidth=0.8, linestyle='-.')
        i += 1

    # 添加方框到坐标轴
    ax.add_patch(rect)
plt.xlabel('Time(s)')
plt.ylabel('Power(W)')
plt.legend()
# plt.show()
plt.savefig('match_event.png', bbox_inches='tight',  dpi=300)
