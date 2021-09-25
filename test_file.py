from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from common import *
from control_math import *

T = 2
t = np.linspace(0, T, SERVO_FREQ * T + 1)
u = np.sin(2 * np.pi * ((3000 - 1000) / T * t ** 2 / 2 + 1000 * t))

f, fw = fft(u, DT)

# plt.figure()
# plt.plot(f, np.abs(fw))
# plt.show()
# signal.correlate(a, v, mode)

a = np.array([1, 2, 3, 4, 0])
b = np.array([5, 6, 7, 8, 0])

Rab = np.zeros_like(a)
for i in range(len(a)):
    Rab[i] = sum(a[0 : -i - 1] * b[i:-1])
print(Rab)
