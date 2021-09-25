import matplotlib.pyplot as plt
import numpy as np
from common import *
from control_math import *
from sympy import solve
from sympy.abc import a, b, c
from scipy import signal

num = np.array([1])
den = np.array([1, 2, 1])
# (-s+2/T)/(s+2/T)
# cross_num = np.convolve(cross_num, [-1, 4 / DT])
# den = np.convolve(den, [1, 4 / DT])

# 2*T/(T*s+2)
num = np.convolve(num, [2 * DT])
den = np.convolve(den, [DT, 2])
G = TransferFunc(num, den, 0.5)

print(G.nom_d)
print(G.den_d)
