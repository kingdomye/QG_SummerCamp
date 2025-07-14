import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

x = np.linspace(0, 200, 50)
y = x
line, = ax.plot(x, y)

# 绘制y = x
def update(frame):
    print(frame)
    line.set_xdata(x[:frame])
    line.set_ydata(y[:frame])
    return line,

ani = FuncAnimation(fig, update, frames=np.arange(0, 50), blit=True)

plt.show()
