import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from common import *
from scipy import signal
from scipy.linalg import toeplitz

# A = np.array([[1, 2], [3, 4]])
# x = np.array([[5], [6]])
# b = np.array([[17], [39]])
# Ai = np.linalg.pinv(A)
# xi = np.linalg.pinv(x)

ps_num, ps_den = (
    np.array(
        [
            5.92592593e01,
            2.60635835e04,
            1.49023716e08,
            3.08684710e10,
            8.31224243e13,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ]
    ),
    np.array(
        [
            1.00000000e00,
            1.70842313e03,
            4.29230033e06,
            4.14332252e09,
            4.51609602e12,
            2.17302741e15,
            7.99662428e17,
            1.11633265e20,
            0.00000000e00,
            0.00000000e00,
        ]
    ),
)
res_freq = 200
anti_res_freq = 150
res_omega = res_freq * 2 * np.pi
anti_res_omega = anti_res_freq * 2 * np.pi

mass = 1
beta1 = 0.1
beta2 = 0.1
res_num = (
    1
    / mass
    * res_omega ** 2
    / anti_res_omega ** 2
    * np.array([1, 2 * beta1 * anti_res_omega, anti_res_omega ** 2])
)
res_den = np.array([1, 2 * beta2 * res_omega, res_omega ** 2])


# process_sensitivity_ir = np.load("process_sensitivity_ir.npy")
# ps_response_mat = np.load("ps_response_mat.npy")
# ps_pinv_mat = np.load("ps_pinv_mat.npy")
# ps_s = fft(process_sensitivity_ir)
# plt.figure()
# plt.plot(20 * np.log10(np.abs(ps_s)))
# ps_inv_s = 1 / ps_s
# ps_inv = np.real(ifft(ps_inv_s))

data_len = 100000
T = data_len / SERVO_FREQ
delta_omega = 2 * np.pi / T

import control as ctrl

ps_tf = ctrl.tf(ps_num, ps_den)
ps_inv_tf = ctrl.tf(ps_den, ps_num)
# ps_tf = ctrl.tf(res_num, res_den)
t, process_sensitivity_ir = signal.impulse(
    (ps_tf.num[0][0], ps_tf.den[0][0]), T=np.arange(DT, T + DT, DT)
)
process_sensitivity_ir = process_sensitivity_ir / SERVO_FREQ

# # omega = delta_omega * np.arange(-int(data_len / 2), int(data_len / 2))
# # omega = np.arange(delta_omega, delta_omega + int(data_len) * delta_omega, delta_omega)
# omega1 = delta_omega * np.arange(1, int(data_len / 2))
# omega2 = delta_omega * np.arange(-int(data_len / 2), 0)
# # ps_bode_point = ps_tf(1j * omega)
# ps_bode_point1 = ps_tf(1j * omega1)
# ps_bode_point2 = ps_tf(1j * omega2)
# ps_bode_point = np.hstack((1e-18j, ps_bode_point1, ps_bode_point2))
# ps_inv_bode_point = 1 / ps_bode_point
# ps_inv_t = np.real(ifft(ps_inv_bode_point))
# ps_t = np.real(ifft(ps_bode_point))

# process_sensitivity_ir_temp = np.roll(process_sensitivity_ir, -1)
ps_inv_t = np.real(ifft(1 / fft(process_sensitivity_ir)))

# ps_response_mat = np.mat(
#     toeplitz(process_sensitivity_ir, np.zeros_like(process_sensitivity_ir)), dtype=float
# )
# ps_pinv_mat = np.linalg.pinv(np.array(ps_response_mat))

# plt.figure()
# plt.plot(t4move, process_sensitivity_ir)


# np.save("process_sensitivity_ir.npy", process_sensitivity_ir)
# np.save("ps_response_mat.npy", ps_response_mat)
# np.save("ps_pinv_mat.npy", ps_pinv_mat)


# plt.figure()
# plt.plot(ps_t, label="ifft")
# plt.plot(process_sensitivity_ir, label="ir")
# plt.figure()
# plt.plot(ps_inv_t, label="ifft")
# # plt.plot(ps_pinv_mat[:, 1], label="pinv")
# # plt.plot(20 * np.log10(np.abs(ps_inv_s)))
# # plt.plot(process_sensitivity_ir)
# plt.legend()
#
# # plt.figure()
# # plt.plot(20 * np.log10(np.abs(ps_bode_point)))
# # plt.figure()
# # plt.plot(np.angle(ps_bode_point, deg=True))
#
# plt.show()
print(ps_inv_t)
