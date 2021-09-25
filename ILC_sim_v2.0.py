from common import *
from control_math import *
from numpy import pi
from scipy.signal import ss2tf, tf2ss

m = 1
k = 1
b = 1

T = 1

A = np.array([[0, 1], [-k / m, -b / m]])
B = np.array([[0], [1 / m]])
C = np.array([1, 0])
D = np.array([0])
# rigid_plant_num, rigid_plant_den = ss2tf(A, B, C, D)
#
# res_freq = 300
# anti_res_freq = 250
# res_omega = res_freq * 2 * np.pi
# anti_res_omega = anti_res_freq * 2 * np.pi
#
# beta1 = 0.1
# beta2 = 0.1
# res_num = (
#     res_omega ** 2
#     / anti_res_omega ** 2
#     * np.array([1, 2 * beta1 * anti_res_omega, anti_res_omega ** 2])
# )
# res_den = np.array([1, 2 * beta2 * res_omega, res_omega ** 2])
#
# num = np.convolve(res_num, rigid_plant_num.flatten())
# den = np.convolve(res_den, rigid_plant_den.flatten())
#
# # A, B, C, D = tf2ss(num[2:], den)
ss = StateSpaceModel(A, B, C, D, DT)

start_freq = 50
end_freq = 5000
start_freq_ = 0.8 * start_freq
end_freq_ = 1.1 * end_freq
T = (
    int(((end_freq_ - start_freq_) / (end_freq - start_freq) * T) * SERVO_FREQ)
    / SERVO_FREQ
)

t = np.linspace(0, T, int(SERVO_FREQ * T) + 1)

u = np.sin(2 * np.pi * ((end_freq_ - start_freq_) / T * t ** 2 / 2 + start_freq_ * t))

# u = np.linspace(0, 1, len(t))
# u = np.sin(2 * pi * 500 * t)

y = np.zeros_like(u)
for i in range(len(u)):
    input_sig = u[i]
    y_output, x_state = ss.response(input_sig, method="linear")
    y[i] = y_output
p = y
y = np.diff(y, 2) / DT / DT
y = np.pad(y, (1, 1), "constant", constant_values=(0, 0))
"""
u_detrend = u - np.mean(u)
y_detrend = y - np.mean(y)

f_u, fw_u = fft(hamm(u_detrend), DT)
f_y, fw_y = fft(hamm(y_detrend), DT)

resolution = 1 / DT / len(f_u)
start_point = int(start_freq / resolution)
end_point = int(end_freq / resolution)

f_u = f_u[start_point:end_point]
fw = fw_y[start_point:end_point] / fw_u[start_point:end_point]"""


plt.figure()
plt.plot(t, u, label="u")
plt.legend()
plt.figure()
plt.plot(t, y, label="y")
plt.legend()

"""plt.figure(figsize=(14, 4))
plt.subplot(121)
plt.xscale("log")
plt.plot(f_u, 20 * np.log10(np.abs(fw)))

plt.subplot(122)
plt.xscale("log")
plt.plot(f_u, np.angle(fw, deg=True))"""
plt.show()
