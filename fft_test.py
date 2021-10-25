import numpy as np

from control_math import *
from common import *

t = np.arange(0, 1, DT)
# u = np.ones_like(t)
# u = np.sin(2 * np.pi * 128 * t)
u = t - 0.5
f, fw = fft(u, DT)
plt.figure()
plt.plot(t, u)

plt.figure()
plt.semilogx(f, np.abs(fw))

plt.figure()
plt.semilogx(f, np.angle(fw, deg=True))


fw_dft = dft_slow(u)
plt.figure()
plt.semilogx(np.abs(fw_dft))

plt.figure()
plt.semilogx(np.angle(fw_dft, deg=True))

sp = np.fft.fft(u)
freq = np.fft.fftfreq(t.shape[-1])
plt.figure()
plt.semilogx(np.abs(sp))
plt.figure()
plt.plot(np.angle(sp, deg=True))
plt.show()

# semilogx
