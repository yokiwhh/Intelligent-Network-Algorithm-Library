import matplotlib.pyplot as plt
import numpy as np

def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')  # numpy的卷积函数

t = np.linspace(start = -4, stop = 4, num = 100)
y = np.sin(t) + np.random.randn(len(t)) * 0.1
y_av = moving_average(interval = y, window_size = 10)
plt.plot(t, y, "b.-", t, y_av, "r.-")

plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(['original data', 'smooth data'])
plt.grid(True)
plt.show()
