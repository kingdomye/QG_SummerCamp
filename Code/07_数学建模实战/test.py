import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 等距螺线参数
p = 55                                          # 螺距
theta = np.linspace(0, 2*np.pi*16, 1000)        # 角度范围
r = theta * p / (2*np.pi)                       # 距离

x = r * np.cos(theta)
y = r * np.sin(theta)

# 画布参数
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_title('龙在等距螺线上移动的动画')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
line, = ax.plot(x, y, color='blue', label='等距螺线')

head_length = 341 - 27.5                        # 龙头有效长度
body_length = 341 - 27.5 * 2                    # 龙身有效长度

# 计算两点之间的距离(极坐标形式)
def calculate_distance(theta1, theta2):
    r1, r2 = theta1 * p / (2*np.pi), theta2 * p / (2*np.pi)
    x1, y1 = r1 * np.cos(theta1), r1 * np.sin(theta1)
    x2, y2 = r2 * np.cos(theta2), r2 * np.sin(theta2)
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

# 找到曲线上的下一个点
def find_next_point(current_theta, theta_range, mode=None):
    theta_sequence = np.linspace(theta_range[0]-current_theta, theta_range[1]-current_theta, 100)

    for i in range(len(theta_sequence)-1):
        theta1 = current_theta + theta_sequence[i]
        theta2 = current_theta + theta_sequence[i+1]
        distance1 = calculate_distance(current_theta, theta1)
        distance2 = calculate_distance(current_theta, theta2)

        if mode == 'body':
            if distance1 <= body_length <= distance2:
                error = abs(distance1 - body_length)
                if error < 0.10:
                    return theta1
                return find_next_point(current_theta, (theta1, theta2), 'body')
        elif mode == 'head':
            if distance1 <= head_length <= distance2:
                error = abs(distance1 - head_length)
                if error < 0.10:
                    return theta1
                return find_next_point(current_theta, (theta1, theta2), 'head')
        else:
            raise ValueError("mode must be 'body' or 'head'")

# 根据已知龙头位置得到所有点坐标
def get_all_points(head_theta, body_nums):
    theta_list = [head_theta]
    r = head_theta * p / (2*np.pi)
    x_start, y_start = r * np.cos(head_theta), r * np.sin(head_theta)
    x, y = [x_start], [y_start]

    theta_first = find_next_point(head_theta, (head_theta, head_theta + np.pi), mode='head')
    theta_list.append(theta_first)
    new_x, new_y = theta_first * p / (2*np.pi) * np.cos(theta_first), theta_first * p / (2*np.pi) * np.sin(theta_first)
    x.append(new_x)
    y.append(new_y)
    
    for _ in range(body_nums):
        new_theta = find_next_point(theta_list[-1], (theta_list[-1], theta_list[-1] + np.pi), mode='body')
        theta_list.append(new_theta)
        new_x, new_y = new_theta * p / (2*np.pi) * np.cos(new_theta), new_theta * p / (2*np.pi) * np.sin(new_theta)
        x.append(new_x)
        y.append(new_y)

    return x, y

# 初始化动画
dragon_line, = ax.plot([], [], color='red', linewidth=2, label='龙')
dragon_head, = ax.plot([], [], 'o', color='red', markersize=8, label='龙头')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    dragon_line.set_data([], [])
    dragon_head.set_data([], [])
    time_text.set_text('')
    return dragon_line, dragon_head, time_text

# 更新函数
def update(frame):
    start_theta = 2 * np.pi * 16
    v = 1
    omega = (2 * np.pi * v) / p
    step = np.linspace(0, -300, 300)
    current_step = step[frame] * omega
    current_theta = start_theta + current_step
    
    x_dragon, y_dragon = get_all_points(current_theta, 221)
    
    dragon_line.set_data(x_dragon, y_dragon)
    dragon_head.set_data(x_dragon[0], y_dragon[0])
    
    # 计算并显示当前时间（秒）
    current_time = 300 * (frame / 300)
    time_text.set_text(f'时间: {current_time:.1f}秒')
    
    return dragon_line, dragon_head, time_text

# 创建动画
ani = FuncAnimation(fig, update, frames=range(300), init_func=init, blit=True, interval=50)

# 添加图例
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# 如果需要保存动画，取消下面一行的注释
# ani.save('dragon_animation.gif', writer='pillow', fps=20)    