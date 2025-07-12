import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置中文字体
plt.rcParams["font.family"] = ["Heiti TC", "STHeiti", "sans-serif"]

# 动画参数
start_x = 10  # 起始点的x坐标
total_frames = 200  # 动画总帧数

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(8, 8))
line, = ax.plot([], [], lw=2, color='blue')
trace, = ax.plot([], [], lw=1, color='gray', alpha=0.5)
title = ax.set_title('顺时针旋入螺旋线动画')

# 设置坐标轴范围
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')

# 初始化函数
def init():
    line.set_data([], [])
    trace.set_data([], [])
    title.set_text('顺时针旋入螺旋线动画')
    return line, trace, title

# 更新函数
def update(frame):
    # 计算当前帧的螺旋线参数
    # 螺旋线从外到内，theta从0递减到-10π
    theta = np.linspace(0, -10 * np.pi * (frame / total_frames), 1000)
    
    # 调整参数使螺旋线从 (start_x, 0) 开始并旋入原点
    a = start_x  # 初始半径
    b = a / (10 * np.pi)  # 控制螺旋线间距的参数，确保螺旋线最终到达原点
    
    # 等距螺旋线的参数方程: r = a + b*theta (theta为负值，从0递减)
    r = a + b * theta
    
    # 转换为笛卡尔坐标 (注意顺时针旋转需要交换sin和cos，并调整符号)
    x = r * np.cos(-theta)  # 负号使旋转方向变为顺时针
    y = r * np.sin(-theta)
    
    # 更新线条数据
    line.set_data(x[-100:], y[-100:])  # 只显示最近的100个点，让动画更流畅
    trace.set_data(x, y)  # 完整轨迹
    
    # 更新标题显示当前状态
    # progress = (frame / total_frames) * 100
    # title.set_text(f'顺时针旋入螺旋线 - 进度: {progress:.1f}%')
    
    return line, trace, title

# 创建动画
ani = FuncAnimation(fig, update, frames=np.linspace(0, total_frames, total_frames+1),
                    init_func=init, blit=True, interval=50)

# 显示动画
plt.tight_layout()
plt.show()

# 如果需要保存动画，取消下面一行的注释
# ani.save('spiral_inward.gif', writer='pillow', fps=20)