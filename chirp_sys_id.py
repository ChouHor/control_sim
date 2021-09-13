import matplotlib.pyplot as plt
import numpy as np
from common import *
from control_math import *
from sympy import solve
from sympy.abc import a, b, c
from scipy import signal


res_freq = 30
anti_res_freq = 20
res_omega = res_freq * 2 * np.pi
anti_res_omega = anti_res_freq * 2 * np.pi

beta1 = 0.1
beta2 = 0.1

num = (
    res_omega ** 2
    / anti_res_omega ** 2
    * np.array([1, 2 * beta1 * anti_res_omega, anti_res_omega ** 2])
)
den = np.array([1, 2 * beta2 * res_omega, res_omega ** 2])

# (-s+2/T)/(s+2/T)
num = np.convolve(num, [-1, 4 / DT])
den = np.convolve(den, [1, 4 / DT])

# 2*T/(T*s+2)
# num = np.convolve(num, [2 * DT])
# den = np.convolve(den, [DT, 2])

# 1/(s+1)
# num = np.convolve(num, [1])
# den = np.convolve(den, [1, 1])

plant_acc_tf = TransferFunc(num, den, DT)

# plant_acc_tf = TransferFunc([0.045, 0.4, 2200], [0.000675, 0.024, 132], DT)
a2p_tf = TransferFunc([1], [1, 0, 0], DT)
plant_pos_tf = a2p_tf * plant_acc_tf

# plant_acc_tf.bode(1, 1000)

# system identification
f, fw_acc = chirp_iden(plant_acc_tf, 5, 1000, 0.5, plot=True)

f, fw_a2p = a2p_tf.bode(f)

fw_pos = fw_acc * fw_a2p
