import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz

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

# controller_tf_model = TransferFunc([50 / np.pi, 100], [1 / 600 / np.pi, 1], DT)
# controller_tf_model.bode(np.arange(1, 2000), plot=True)

identity_tf = TransferFunc([1], [1], DT)
closed_loop_tf_model = (
    controller_tf_model
    * plant_pos_tf
    / (identity_tf + controller_tf_model * plant_pos_tf)
)

process_sensitivity_tf_model = plant_pos_tf / (
    identity_tf + controller_tf_model * plant_pos_tf
)

ps_inv_tf_model = identity_tf / process_sensitivity_tf_model
# process_sensitivity_tf_model.bode(np.arange(1, 200), plot=True)
# aa
# SPG
T = 0.1
y1 = 1
sol = solve(
    [
        a * T ** 5 + b * T ** 4 + c * T ** 3 - y1,
        5 * a * T ** 4 + 4 * b * T ** 3 + 3 * c * T ** 2,
        20 * a * T ** 3 + 12 * b * T ** 2 + 6 * c * T,
    ],
    [a, b, c],
)

t = np.linspace(0, T, int(SERVO_FREQ * T))
# t2 = np.append(t4dyn, np.zeros(int(SERVO_FREQ * T4chirp / 2)))
set_point = sol[a] * t ** 5 + sol[b] * t ** 4 + sol[c] * t ** 3
# set_point = np.append(set_point, np.ones(int(SERVO_FREQ * T4chirp / 2)))
# t4dyn = t2

# get unit response
unit_input = np.zeros_like(t)
unit_input[0] = 1
plant_unit_response = np.array([])
# get plant response
for i in range(len(unit_input)):
    input_sig = unit_input[i]
    p_current = plant_pos_tf.response(input_sig)  # +random.normal()/4/5
    plant_unit_response = np.append(plant_unit_response, p_current)
plant_unit_response = np.array(plant_unit_response)
fig, axs = plt.subplots()
axs.plot(plant_unit_response, label="plant unit response")
axs.legend(loc="upper right")
plant_response_mat = np.mat(
    toeplitz(plant_unit_response, np.zeros_like(plant_unit_response)), dtype=float
)
plant_pinv_mat = np.linalg.pinv(plant_response_mat)

# get process sensitivity response
ps_unit_response = np.array([])
for i in range(len(unit_input)):
    input_sig = unit_input[i]
    p_current = process_sensitivity_tf_model.response(input_sig)  # +random.normal()/4/5
    ps_unit_response = np.append(ps_unit_response, p_current)
ps_unit_response = np.array(ps_unit_response)
fig, axs = plt.subplots()
axs.plot(ps_unit_response, label="process sensitivity unit response")
axs.legend(loc="upper right")
ps_response_mat = np.mat(
    toeplitz(ps_unit_response, np.zeros_like(ps_unit_response)), dtype=float
)
ps_pinv_mat = np.linalg.pinv(ps_response_mat)

A = np.identity(ps_response_mat.shape[0]) - ps_pinv_mat * ps_response_mat
U, sigma, VT = np.linalg.svd(A)
print(max(sigma))


Q_num, Q_den = signal.butter(4, 2000, "low", analog=True)
# Q_num, Q_den = ([1], [1])
Q_tf1 = TransferFunc(Q_num, Q_den, DT)
Q_tf2 = TransferFunc(Q_num, Q_den, DT)

# iteration
current_ILC = np.zeros_like(set_point)
next_ILC = np.zeros_like(set_point)
for k in range(1):
    pos = np.array([0])
    err = np.array([])
    pid_output = np.array([])

    current_ILC = next_ILC
    Q_tf1.reset()
    Q_tf2.reset()
    plant_pos_tf.reset()
    controller_tf_model.reset()
    for i in range(len(set_point)):
        input_sig = set_point[i]
        current_err = input_sig - pos[-1]
        err = np.append(err, current_err)
        current_pid_output = controller_tf_model.response(current_err)
        pid_output = np.append(pid_output, current_pid_output)

        plant_input = current_pid_output + current_ILC[i]
        p_current = plant_pos_tf.response(plant_input)  # +random.normal()/4/5
        pos = np.append(pos, p_current)

    err = np.array(err, dtype=float)
    pid_output = np.array(pid_output, dtype=float)

    # current plant in as next ffc
    # Le = np.zeros_like(set_point)
    # Le_after_Q = np.zeros_like(set_point)
    # current_ILC_after_Q = np.zeros_like(set_point)
    # next_ILC = current_ILC + pid_output

    # inverse plant
    # Le = np.array(plant_pinv_mat * err.reshape(-1, 1)).reshape(-1)
    # Le_after_Q = np.zeros_like(set_point)
    # current_ILC_after_Q = np.zeros_like(set_point)
    # next_ILC = current_ILC + pid_output + Le

    # inverse ps (paper method)
    # Le = np.array(ps_pinv_mat * err.reshape(-1, 1)).reshape(-1)

    Le = np.array([])
    for i in range(len(set_point)):
        input_sig = err[i]
        current_Le = ps_inv_tf_model.response(input_sig)
        Le = np.append(Le, current_Le)
    Le = np.array(Le, dtype=float)

    # without Q
    Le_after_Q = np.zeros_like(set_point)
    current_ILC_after_Q = np.zeros_like(set_point)
    next_ILC = current_ILC + Le

    # with Q
    # next_ILC_input_before_Q = current_ILC + Le
    # current_ILC_after_Q = np.array([])
    # Le_after_Q = np.array([])
    # next_ILC = np.array([])
    #
    # for i in range(len(set_point)):
    #     # input_sig = next_ILC_input_before_Q[i]
    #     # current_after_Q = Q_tf.response(input_sig)
    #     # next_ILC = np.append(next_ILC, current_after_Q)
    #     input_sig = current_ILC[i]
    #     current_current_ILC_after_Q = Q_tf1.response(input_sig)
    #     current_ILC_after_Q = np.append(
    #         current_ILC_after_Q, current_current_ILC_after_Q
    #     )
    #     input_sig = Le[i]
    #     current_Le_after_Q = Q_tf2.response(input_sig)
    #     Le_after_Q = np.append(Le_after_Q, current_Le_after_Q)
    #
    # current_ILC_after_Q = np.array(current_ILC_after_Q, dtype=float)
    # Le_after_Q = np.array(Le_after_Q, dtype=float)
    # next_ILC = current_ILC_after_Q + Le_after_Q

    pos = np.array(pos, dtype=float)[1:]
    # plt.figure(figsize=(6, 5))
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Iteration " + str(k + 1))
    axs[0, 0].plot(set_point, label="cmd")
    axs[0, 0].plot(pos, label="pos")
    axs[0, 0].legend(loc="upper right")
    axs[0, 1].plot(err, label="err")
    axs[0, 1].legend(loc="upper right")
    axs[1, 0].plot(pid_output, label="pid out")
    axs[1, 0].legend(loc="upper right")
    axs[1, 1].plot(current_ILC, label="f_k")
    axs[1, 1].legend(loc="upper right")
    axs[2, 0].plot(Le, label="Le b4 Q")
    axs[2, 0].legend(loc="upper right")
    axs[2, 1].plot(Le_after_Q, label="Le aft Q")
    axs[2, 1].legend(loc="upper right")
    axs[3, 0].plot(current_ILC_after_Q, label="f_k aft Q")
    axs[3, 0].legend(loc="upper right")
    axs[3, 1].plot(next_ILC, label="f_k+1 aft Q")
    axs[3, 1].legend(loc="upper right")
    plt.show()
