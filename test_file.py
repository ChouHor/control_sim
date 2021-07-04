from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from common import *
from control_math import *

res_freq = 200
anti_res_freq = 150
res_omega = res_freq * 2 * np.pi
anti_res_omega = anti_res_freq * 2 * np.pi

beta1 = 0.1
beta2 = 0.1

# plant_pos_tf = TransferFunc([1], [1, 2, 1], DT)
plant_pos_tf = TransferFunc(
    res_omega ** 2
    / anti_res_omega ** 2
    * np.array([1, 2 * beta1 * anti_res_omega, anti_res_omega ** 2]),
    np.array([1, 2 * beta2 * res_omega, res_omega ** 2]),
    DT,
)
t = np.linspace(0, 7, 7000)
u1 = np.zeros_like(t)
u1[0] = SERVO_FREQ
impulse_response = np.array([0])

for i in range(len(u1)):
    input_sig = u1[i]
    p_current = plant_pos_tf.response(input_sig)  # +random.normal()/4/5
    impulse_response = np.append(impulse_response, p_current)

p = np.array(impulse_response)  # [1:]
plt.figure(1)
plt.plot(p)

# signal lsim method
# plant_pos_tf = signal.lti([1], [1, 2, 1])
plant_pos_tf = signal.lti(
    res_omega ** 2
    / anti_res_omega ** 2
    * np.array([1, 2 * beta1 * anti_res_omega, anti_res_omega ** 2]),
    np.array([1, 2 * beta2 * res_omega, res_omega ** 2]),
)

t = np.linspace(0, 7, 7000)
u = np.zeros_like(t)
u[0] = 1
tout1, y1, x1 = signal.lsim(plant_pos_tf, u, t)
plt.figure(2)
plt.plot(tout1, y1)

# signal impulse method
tout2, y2 = signal.impulse2(plant_pos_tf, T=t)
# tout2, y2 = signal.step(plant_pos_tf, T=t)
plt.figure()
plt.plot(tout2, y2)

# plt.plot(t, y)


# plt.figure(figsize=(12, 4))
# plt.subplot(121)
# plt.xscale("log")
# plt.plot(
#     f,
#     20 * np.log10(np.abs(fw_acc)),
# )
#
# plt.subplot(122)
# plt.xscale("log")
# plt.plot(
#     f,
#     np.angle(fw_acc, deg=True),
# )
# plt.show()
