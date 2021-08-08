import matplotlib.pyplot as plt
import numpy as np
from common import *
from control_math import *
from sympy import solve
from sympy.abc import a, b, c
from scipy import signal


res_freq = 200
anti_res_freq = 150
res_omega = res_freq * 2 * np.pi
anti_res_omega = anti_res_freq * 2 * np.pi

beta1 = 0.1
beta2 = 0.1

plant_acc_tf = TransferFunc(
    res_omega ** 2
    / anti_res_omega ** 2
    * np.array([1, 2 * beta1 * anti_res_omega, anti_res_omega ** 2]),
    np.array([1, 2 * beta2 * res_omega, res_omega ** 2]),
    DT,
)

# plant_acc_tf = TransferFunc([0.045, 0.4, 2200], [0.000675, 0.024, 132], DT)
a2p_tf = TransferFunc([1], [1, 0, 0], DT)
plant_pos_tf = a2p_tf * plant_acc_tf

# plant_acc_tf.bode(np.arange(1, 250), plot=False)

# # system identification
# f, fw_acc = chirp_iden(plant_acc_tf, 1, 250, 1, plot=False)
#
# f, fw_a2p = a2p_tf.bode(f)
#
# fw_pos = fw_acc * fw_a2p

# # Controller parameters
kp, ki, kd = pole_placement(0, 10, 0, SERVO_FREQ)
# kp = 100000
# ki = 0.0
# kd = 0

controller_kp = TransferFunc([kp], [1], DT)
controller_ki = TransferFunc([ki], [1, 0], DT)
controller_kd = TransferFunc([kd, 0], [1], DT)
controller_tf_model = controller_kp + controller_ki + controller_kd
# controller_tf_model.bode(np.array(range(10, 5000)), plot=True)

# controller_tf_model = TransferFunc([50 / np.pi, 100], [1 / 600 / np.pi, 1], DT)
# controller_tf_model.bode(np.arange(1, 2000), plot=True)

identity_tf = TransferFunc([1], [1], DT)
open_loop_tf_model = controller_tf_model * plant_pos_tf
# open_loop_tf_model.bode(np.array(range(10, 1000)), plot=True)

closed_loop_tf_model = (
    controller_tf_model
    * plant_pos_tf
    / (identity_tf + controller_tf_model * plant_pos_tf)
)
# closed_loop_tf_model.bode(np.array(range(10, 5000)), plot=True)

process_sensitivity_tf_model = plant_pos_tf / (
    identity_tf + controller_tf_model * plant_pos_tf
)

ps_inv_tf_model = identity_tf / process_sensitivity_tf_model

# ps_inv_tf_model.zpk()
# nom = ps_inv_tf_model.nom
# den = ps_inv_tf_model.den
# z, p, k = signal.tf2zpk(nom, den)

"""Simulation"""
T = 0.1
y1 = 0.1
sol = solve(
    [
        a * T ** 5 + b * T ** 4 + c * T ** 3 - y1,
        5 * a * T ** 4 + 4 * b * T ** 3 + 3 * c * T ** 2,
        20 * a * T ** 3 + 12 * b * T ** 3 + 6 * c * T,
    ],
    [a, b, c],
)

t = np.linspace(0, T, int(SERVO_FREQ * T))
# t2 = np.append(t, np.zeros(int(SERVO_FREQ * T / 2)))
set_point = sol[a] * t ** 5 + sol[b] * t ** 4 + sol[c] * t ** 3
# set_point = np.append(set_point, np.ones(int(SERVO_FREQ * T / 2)))
# t = t2

pos = np.array([0], dtype=float)
vel = np.array([0], dtype=float)
err = np.array([], dtype=float)
ctrl = np.array([], dtype=float)

for i in range(len(set_point)):
    input_sig = set_point[i]
    # 分开算
    # err_current = input_sig - pos[-1]
    # ctrl_current = controller_tf_model.response(err_current)
    # pos_current = plant_pos_tf.response(ctrl_current)  # +random.normal()/4/5
    # ctrl = np.append(ctrl, ctrl_current)
    # 分开积分算
    # err_current = input_sig - pos[-1]
    # ctrl_current = controller_tf_model.response(err_current)
    # acc_current = plant_acc_tf.response(ctrl_current)  # +random.normal()/4/5
    # vel_current = vel[-1] + acc_current * DT
    # pos_current = pos[-1] + vel_current * DT

    # ctrl = np.append(ctrl, ctrl_current)
    # 一起算
    pos_current = closed_loop_tf_model.response(input_sig)  # +random.normal()/4/5

    # vel = np.append(vel, vel_current)
    pos = np.append(pos, pos_current)

pos = np.delete(pos, 0)
err = set_point - pos
fig, axes = plt.subplots(1, 3, figsize=(21, 4))

axes[0].set_xlabel("time / [s]")
axes[0].set_ylabel("Cmd / [N]")
axes[0].plot(t, set_point, label="Cmd")
axes[0].plot(t, pos, label="Pos")
axes[0].legend()

axes[1].set_xlabel("time / [s]")
axes[1].set_ylabel("Err / [m/s^2]")
axes[1].plot(t, err)

axes[2].set_xlabel("time / [s]")
axes[2].set_ylabel("Err / [m/s^2]")
axes[2].plot(t, ctrl)

plt.suptitle("Move")
plt.show()

# Le = np.array([0], dtype=float)
#
# for i in range(len(err)):
#     input_sig = err[i]
#     Le_current = ps_inv_tf_model.response(input_sig)  # +random.normal()/4/5
#     Le = np.append(Le, Le_current)
#
# Le = np.delete(Le, 0)
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 4))
#
# axes[0].set_xlabel("time / [s]")
# axes[0].set_ylabel("Err / [N]")
# axes[0].plot(t, err, label="Err")
# axes[0].legend()
#
# axes[1].set_xlabel("time / [s]")
# axes[1].set_ylabel("Le / [m/s^2]")
# axes[1].plot(t, Le)
#
# plt.suptitle("Move")
# plt.show()
