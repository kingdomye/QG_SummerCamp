import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置中文字体
plt.rcParams["font.family"] = ["Heiti TC", "STHeiti", "sans-serif"]

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(8, 8))
line, = ax.plot([], [], lw=2, color='blue')

ax.set_aspect('equal')
ax.grid(True)
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')

total_frames = 1000

# 计算最大半径，用于设置坐标轴范围
max_theta = 16 * 2 * np.pi
p = 55
max_r = max_theta * p / (2 * np.pi)

# 设置坐标轴范围，确保能看到完整的螺线
ax.set_xlim(-max_r * 1.1, max_r * 1.1)
ax.set_ylim(-max_r * 1.1, max_r * 1.1)

# 初始化函数
def init():
    line.set_data([], [])
    return [line]

def update(frame):
    theta = np.linspace(0, max_theta * (frame / total_frames), 1000)
    r = theta * p / (2 * np.pi)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    line.set_data(x, y)
    return [line]

# 创建动画，使用 np.linspace 生成正确的帧序列
ani = FuncAnimation(
    fig, update, 
    frames=np.linspace(0, total_frames, total_frames+1),  # 正确的帧序列
    init_func=init, 
    blit=True,
    interval=50  # 设置动画速度，每帧20毫秒
)

plt.tight_layout()
plt.show()
