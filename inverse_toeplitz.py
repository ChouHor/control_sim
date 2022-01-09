import numpy as np
import scipy.signal

from common import *
from control_math import *
from numpy import pi
from scipy.signal import ss2tf, tf2ss, impulse, impulse2, step, butter, lsim, lsim2
from sympy import solve
from sympy.abc import a, b, c
from scipy.linalg import toeplitz


def inverse_toeplitz(an):
    bn = np.zeros_like(an, dtype=float)
    a0 = an[0]
    bn[0] = 1 / a0
    an_flip = np.flip(an)
    for i in range(1, len(bn)):
        bn[i] = -sum(bn[0:i] * an_flip[-i - 1 : -1]) / a0
    return bn


# ir = np.array([5, 2, 3, 4])
ir = np.random.random(1000)
ir_mat = np.mat(toeplitz(ir, np.zeros_like(ir)), dtype=float)
ir_pinv_mat = np.linalg.pinv(ir_mat)
print(ir_pinv_mat[:, 0])
bn = inverse_toeplitz(ir)
print(bn)
print(np.mat(toeplitz(ir, np.zeros_like(ir)), dtype=float).dot(bn))
print(ir_pinv_mat.dot(ir))
