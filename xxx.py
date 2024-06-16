import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义y的范围和步长
y_min, y_max = -5, 5
y_step = 1
y_values = np.arange(y_min, y_max + y_step, y_step)

# 定义每条折线的x位置（固定值）和z关于y的函数
x_positions = [1, 2, 3]  # x坐标位置
functions = [lambda y: y**2, lambda y: y**3, lambda y: np.sin(y)]  # z关于y的函数列表

# 初始化存储所有点的列表
all_x = []
all_y = []
all_z = []

# 计算每条折线的点
for x_pos, func in zip(x_positions, functions):
    z_values = func(y_values)
    all_x.extend([x_pos] * len(y_values))
    all_y.extend(y_values)
    all_z.extend(z_values)

# 将列表转换为NumPy数组以便绘图
all_x = np.array(all_x)
all_y = np.array(all_y)
all_z = np.array(all_z)

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制每条折线
for i in range(len(x_positions)):
    start_idx = i * len(y_values)
    end_idx = (i + 1) * len(y_values)
    ax.plot(all_x[start_idx:end_idx], all_y[start_idx:end_idx], all_z[start_idx:end_idx], label=f'Line {i+1}')

# 设置坐标轴标签
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# 设置图例
ax.legend()

# 调整视角以更好地查看图形
ax.view_init(elev=90., azim=0.)

# 显示网格（可选）
ax.grid(True)

# 显示图形
plt.show()