import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

p = 55                                          # 螺距
theta = np.linspace(0, 2*np.pi*20, 1000)
r = theta * p / (2*np.pi)

x = r * np.cos(theta)
y = r * np.sin(theta)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(x, y, color='blue')

start_theta = 16 * 2 * np.pi
start_r = start_theta * p / (2*np.pi)
start_x, start_y = start_r * np.cos(start_theta), start_r * np.sin(start_theta)
x, y = [], []
head_length = 341 - 27.5                        # 龙头有效长度
body_length = 341 - 27.5 * 2                    # 龙身有效长度

ax.plot(start_x, start_y, 'ro')

plt.show()
