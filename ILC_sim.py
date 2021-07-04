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

sys = TransferFunc(
    res_omega ** 2
    / anti_res_omega ** 2
    * np.array([1, 2 * beta1 * anti_res_omega, anti_res_omega ** 2]),
    np.array([1, 2 * beta2 * res_omega, res_omega ** 2]),
    DT,
)
sys.bode_plot(1, 2000)
chirp_iden(sys, 10, 2000, 1)

# # Controller parameters
# kp = 1000
# ki = 0.1
# kd = 1000
#
# controller_kp = TransferFunc([kp], [1], DT)
# controller_ki = TransferFunc([ki], [1, 0], DT)
# controller_kd = TransferFunc([kd, 0], [1], DT)
# controller_tf_model = controller_kp + controller_ki + controller_kd
#
# identity_tf = TransferFunc([1], [1], DT)
# closed_loop_tf_model = (
#     controller_tf_model * sys / (identity_tf + controller_tf_model * sys)
# )
#
# x1 = 1
# y1 = 1
# sol = solve(
#     [
#         a * x1 ** 5 + b * x1 ** 4 + c * x1 ** 3 - y1,
#         5 * a * x1 ** 4 + 4 * b * x1 ** 3 + 3 * c * x1 ** 2,
#         20 * a * x1 ** 3 + 12 * b * x1 ** 3 + 6 * c * x1,
#     ],
#     [a, b, c],
# )
# T = 1
# t = np.linspace(0, T, SERVO_FREQ * T)
# set_point = sol[a] * t ** 5 + sol[b] * t ** 4 + sol[c] * t ** 3
#
# p = np.array([0])
# for i in range(len(set_point)):
#     input_sig = set_point[i]
#     err = input_sig - p[-1]
#     ctrl_output = controller_tf_model.response(err)
#     p_current = sys.response(ctrl_output)  # +random.normal()/4/5
#     p = np.append(p, p_current)
#
#
# p = p[1:]
# plt.figure(1)
# plt.plot(p)
# plt.figure(2)
# plt.plot(set_point)
# plt.figure(3)
# plt.plot(set_point - p)
# plt.show()
