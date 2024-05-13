from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

ell1 = Ellipse(xy = (0.0, 0.0), width = 8, height = 4, facecolor= 'yellow', alpha=0.3, fill=False)
ax.add_patch(ell1)

x, y = 0, 0
#ax.plot(x, y, 'ro')

plt.axis('scaled')

plt.axis('equal')   #changes limits of x or y axis so that equal increments of x and y have the same length

plt.show()