import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 等距螺线参数
p = 55                                          # 螺距
theta = np.linspace(0, 2*np.pi*16, 1000)        # 角度范围
r = theta * p / (2*np.pi)                       # 距离

init_x = r * np.cos(theta)
init_y = r * np.sin(theta)

# 画布参数
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)

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
            return ValueError("mode must be 'body' or 'head'")


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


# 碰撞检测
def check_collision(coordinates_p, coordinates_rectangle)->bool:
    x_p, y_p = coordinates_p
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = coordinates_rectangle[0], coordinates_rectangle[1], coordinates_rectangle[2], coordinates_rectangle[3]

    vector_rectangle_1 = np.array([x2 - x1, y2 - y1])
    vector_rectangle_2 = np.array([x3 - x1, y3 - y1])
    # 向量叉乘求面积
    S_rectangle = abs(np.cross(vector_rectangle_1, vector_rectangle_2))

    vector1 = np.array([x_p - x1, y_p - y1])
    vector2 = np.array([x_p - x2, y_p - y2])
    vector3 = np.array([x_p - x3, y_p - y3])
    vector4 = np.array([x_p - x4, y_p - y4])

    S1 = abs(np.cross(vector1, vector2))/2
    S2 = abs(np.cross(vector2, vector3))/2
    S3 = abs(np.cross(vector3, vector4))/2
    S4 = abs(np.cross(vector4, vector1))/2

    S = S1 + S2 + S3 + S4

    if abs(S - S_rectangle) < 0.1:
        return True
    return False


def update(frame):
    ax.clear()
    ax.set_aspect('equal')
    ax.plot(init_x, init_y, color='blue')
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    
    theta = start_theta - frame * omega
    x, y = get_all_points(theta, 221)

    ax.plot(x, y, color='red')
    ax.scatter(x, y, color='red')
    return ax


if __name__ == '__main__':
    start_theta = 2 * np.pi * 16
    v = 3
    omega = (2 * np.pi * v) / p

    frame = np.linspace(0, 300, 300)
    ani = FuncAnimation(
        fig, update, frames=frame, interval=30
    )

    plt.show()
    # coordinates_p = [0, 0]
    # coordinates_rectangle = [(0, 0), (100, 0), (100, 100), (0, 100)]
    # check_collision(coordinates_p, coordinates_rectangle)
