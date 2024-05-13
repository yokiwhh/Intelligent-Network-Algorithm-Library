import numpy as np
from scipy.signal import medfilt

# 生成带有振荡的数据
t = np.linspace(0, 1, 200)
data = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) + 0.2*np.random.randn(200)

# 多次使用中值滤波器平滑数据
filtered_data = data
for i in range(5):
    filtered_data = medfilt(filtered_data, kernel_size=5)

# 绘制结果
import matplotlib.pyplot as plt
plt.plot(t, data, label='原始数据')
plt.plot(t, filtered_data, label='平滑后数据')
plt.legend()
plt.show()