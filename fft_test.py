from control_math import *
from common import *

t = np.arange(0, 2, DT)
# u = np.sin(2 * np.pi * 128 * t)
u = t
f, fw = fft(u, DT)
plt.figure()
plt.plot(t, u)
plt.show()
plt.figure()
plt.semilogx(f, np.abs(fw))
plt.show()
plt.figure()
plt.semilogx(f, np.angle(fw, deg=True))
plt.show()

fw_dft = dft_slow(u)
plt.figure()
plt.semilogx(np.abs(fw_dft))
plt.show()
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
