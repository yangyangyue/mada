from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rc("font", family="Times New Roman")

# 定义每条折线的x位置（固定值）和z关于y的函数
x_positions = [1, 1.5, 2, 2.5, 3, 3.5]  # x坐标位置

lines = []
# 初始化存储所有点的列表
for app_abb in "kmdwf":
    lines.append(np.load(Path('examples') / f'refit2-{app_abb}.npy')[0])
agg = np.stack(lines).sum(axis=0) + 10
lines = [agg] + lines

labels = ['mains', 'kettle', 'microwave', 'dishwasher', 'washing-machine', 'fridge']
# default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# new_colors = default_colors[1:]  
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=new_colors)
# labels = ['kettle', 'microwave', 'dishwasher', 'washing-machine', 'fridge']

# 创建3D图形
fig = plt.figure(figsize=(12, 12))
ax = fig.add_axes(111, projection='3d')

# 绘制每条折线
for i in range(len(lines)):
    line = lines[i]
    ax.plot(np.array(range(len(line))),  [x_positions[i]] * len(line), line, label=labels[i])

# 设置坐标轴标签
ax.set_xlabel('Time(8s)')
ax.set_zlabel('Power(W)')
ax.set_yticks(x_positions)
ax.set_yticklabels(labels)

plt.gca().invert_xaxis()

ax.grid(False)

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # 白色，完全透明
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # 白色，完全透明
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # 白色，不透明（通常是背景，所以设置为不透明）

# # 设置图例
# ax.legend()

# 调整视角以更好地查看图形
ax.view_init(elev=18., azim=54)

ax.set_proj_type('ortho')

plt.savefig('kmdwf.png', bbox_inches='tight', pad_inches=0.5, dpi=300)

# 显示图形
# plt.show()