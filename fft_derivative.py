import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, fftfreq

n = 64
L = 30
dx = L / n
# x = np.arange(-L / 2, L / 2, dx)
x = np.arange(0, L, dx)
f = np.sin(x)
# fhat = np.fft.fft(f)
fhat = fft(f)
fft_freq = fftfreq(n, d=dx) * 2 * np.pi
kappa = (2 * np.pi / L) * np.arange(-n / 2, n / 2)
# kappa = (2 * np.pi / L) * np.arange(0, n)
# kappa = np.fft.fftshift(kappa)
kappa = fftshift(kappa)
dfhat = kappa * fhat * (1j)
# dfFFt = np.real(np.fft.ifft(dfhat))
dfFFt = np.real(ifft(dfhat))

plt.figure()
plt.plot(x, dfFFt)
plt.show()
