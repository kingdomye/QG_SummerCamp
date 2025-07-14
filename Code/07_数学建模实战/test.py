import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置中文字体
plt.rcParams["font.family"] = ["Heiti TC", "STHeiti", "sans-serif"]

total_frames = 200  # 动画总帧数
frames = np.linspace(0, total_frames, total_frames+1)
frames /= total_frames

print(frames)
 