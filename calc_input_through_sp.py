import matplotlib.pyplot as plt
import numpy as np
from common import *
from control_math import *
from sympy import solve
from sympy.abc import a, b, c


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

# plant_acc_tf.bode(1, 1000)

# system identification
# f, fw_acc = chirp_iden(plant_acc_tf, 1, 5000, 0.5)
#
# f, fw_a2p = a2p_tf.bode(f)
#
# fw_pos = fw_acc * fw_a2p


# # Controller parameters
kp = 1000
ki = 0.1
kd = 1000

controller_kp = TransferFunc([kp], [1], DT)
controller_ki = TransferFunc([ki], [1, 0], DT)
controller_kd = TransferFunc([kd, 0], [1], DT)
controller_tf_model = controller_kp + controller_ki + controller_kd
# controller_tf_model.bode(np.arange(1, 2000), plot=True)


identity_tf = TransferFunc([1], [1], DT)
closed_loop_tf_model = (
    controller_tf_model
    * plant_pos_tf
    / (identity_tf + controller_tf_model * plant_pos_tf)
)

# SPG
T = 0.1
y1 = 2e-5
sol = solve(
    [
        a * T ** 5 + b * T ** 4 + c * T ** 3 - y1,
        5 * a * T ** 4 + 4 * b * T ** 3 + 3 * c * T ** 2,
        20 * a * T ** 3 + 12 * b * T ** 3 + 6 * c * T,
    ],
    [a, b, c],
)

t = np.linspace(0, T, int(SERVO_FREQ * T))
# set_point = np.array(sol[a] * t4dyn ** 5 + sol[b] * t4dyn ** 4 + sol[c] * t4dyn ** 3, dtype=float)
set_point = np.linspace(0, y1, len(t), dtype=float)

# Closed loop
# Get closed loop unit response
unit_input = np.zeros_like(t)
unit_input[0] = 1

cl_unit_response = np.array([])
for i in range(len(unit_input)):
    input_sig = unit_input[i]
    p_current = closed_loop_tf_model.response(input_sig)  # +random.normal()/4/5
    cl_unit_response = np.append(cl_unit_response, p_current)
cl_unit_response = np.array(cl_unit_response, dtype=float)
fig, axs = plt.subplots()
fig.suptitle("Closed loop unit response")
axs.plot(cl_unit_response)

cl_unit_response_mat = np.mat(
    toeplitz(cl_unit_response, np.zeros_like(cl_unit_response)), dtype=float
)
cl_sp_response = np.array(
    cl_unit_response_mat * set_point.reshape(-1, 1), dtype=float
).reshape(-1)

# 根据 profile 和 单位脉冲响应反算出的输入
cl_unit_pinv_mat = np.linalg.pinv(cl_unit_response_mat)
cl_calc_input4sp = np.array(
    cl_unit_pinv_mat * set_point.reshape(-1, 1), dtype=float
).reshape(-1)

fig, axs = plt.subplots()
fig.suptitle("Closed loop input subject to set point")
axs.plot(cl_calc_input4sp, label="input")
axs.legend(loc="upper right")

fig, axs = plt.subplots(2)
fig.suptitle("Closed loop response calculated by conv(unit response, set point)")
axs[0].plot(set_point, label="cmd")
axs[0].plot(cl_sp_response, label="pos")
axs[0].legend(loc="upper right")
axs[1].plot(set_point - cl_sp_response, label="err")
axs[1].legend(loc="upper right")

pos = np.array([0])
for i in range(len(set_point)):
    input_sig = set_point[i]
    p_current = closed_loop_tf_model.response(input_sig)  # +random.normal()/4/5
    pos = np.append(pos, p_current)

pos = np.array(pos, dtype=float)[1:]
# plt.figure(figsize=(6, 5))
fig, axs = plt.subplots(2)
fig.suptitle("Closed loop response step by step")
axs[0].plot(set_point, label="cmd")
axs[0].plot(pos, label="pos")
axs[0].legend(loc="upper right")
axs[1].plot(set_point - pos, label="err")
axs[1].legend(loc="upper right")
plt.show()

# Open loop
# Get Open loop unit response
ol_unit_response = np.array([])
for i in range(len(unit_input)):
    input_sig = unit_input[i]
    p_current = plant_pos_tf.response(input_sig)  # +random.normal()/4/5
    ol_unit_response = np.append(ol_unit_response, p_current)
ol_unit_response = np.array(ol_unit_response, dtype=float)
fig, axs = plt.subplots()
fig.suptitle("Open loop unit response")
axs.plot(ol_unit_response)

ol_unit_response_mat = np.mat(
    toeplitz(ol_unit_response, np.zeros_like(ol_unit_response)), dtype=float
)

# 根据 profile 和 单位脉冲响应反算出的输入
ol_unit_pinv_mat = np.linalg.pinv(ol_unit_response_mat)
ol_calc_input4sp = np.array(
    ol_unit_pinv_mat * set_point.reshape(-1, 1), dtype=float
).reshape(-1)

fig, axs = plt.subplots()
fig.suptitle("Open loop input subject to set point")
axs.plot(ol_calc_input4sp, label="input")
axs.legend(loc="upper right")

pos = np.array([0])
for i in range(len(ol_calc_input4sp)):
    input_sig = ol_calc_input4sp[i]
    p_current = plant_pos_tf.response(input_sig)  # +random.normal()/4/5
    pos = np.append(pos, p_current)

pos = np.array(pos, dtype=float)[1:]
# plt.figure(figsize=(6, 5))
fig, axs = plt.subplots(2)
fig.suptitle("Open loop response step by step")
axs[0].plot(set_point, label="cmd")
axs[0].plot(pos, label="pos")
axs[0].legend(loc="upper right")
axs[1].plot(set_point - pos, label="err")
axs[1].legend(loc="upper right")
plt.show()
