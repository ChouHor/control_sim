import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from numpy import pi
from mpl_toolkits.mplot3d import Axes3D

# f = np.arange(1, 100, 1)
# delta = np.arange(0, 1, 0.1)
# s = delta + 1j * 2 * pi * f
# Gs = 1 / s
# plt.figure()
# plt.semilogx(f, np.log10(np.abs(Gs)))
# plt.show()

f = np.arange(-1.5, 1.5, 0.02)
delta = np.arange(-10, 10, 0.2)
f, delta = np.meshgrid(f, delta)
z = 1 / (delta + 1j * 2 * pi * f)

fig = plt.figure()
ax = Axes3D(fig)
gain = np.abs(z)
phase = np.angle(z)
ax.plot_surface(f, delta, gain, rstride=1, cstride=1, cmap="jet")
ax.set_zlim3d(0, 6)

fig2 = plt.figure()
ax2 = Axes3D(fig2)
ax2.plot_surface(f, delta, phase, rstride=1, cstride=1, cmap="jet")
ax2.set_zlim3d(0, 6)

plt.show()
