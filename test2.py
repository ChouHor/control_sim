import numpy as np
import matplotlib.pyplot as plt
from common import *


def trapz(f, a, b, N=50):
    x = np.linspace(a, b, N + 1)
    y = f(x)
    y_right = y[1:]
    y_left = y[:-1]
    dx = (b - a) / N
    T = dx / 2 * sum(y_right + y_left)
    return T


def invlt(t, fs, sigma, omiga, nint):
    omigadim = np.linspace(0, omiga, nint + 1, endpoint=True)
    y = [(np.exp(1j * o * t) * fs(sigma + 1j * o)).real for o in omigadim]
    y_left = y[:-1]
    y_right = y[0:]
    T = sum(y_right + y_left) * omiga / nint
    return np.exp(sigma * t) * T / np.pi / 2


# ------------------------------------------------------------
def fs(s):
    return s / (s * s + 1)


# ------------------------------------------------------------
sigma = -0 + 0.05
omiga = 200
nint = omiga * 50

# tdim = np.linspace(0, 2 * np.pi * 3, 200)
tdim = np.linspace(0, 20, 300)
ft = [invlt(t, fs, sigma, omiga, nint) for t in tdim]

plt.plot(ft)
