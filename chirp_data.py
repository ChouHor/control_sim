from control_math import *
from common import *

# Setup Chirp signal
dt = DT
t_total = 0.1
num = t_total / dt
t = np.arange(0, t_total, dt)

f_chirp_start = 1
f_chirp_end = 2000

f_inspect_start = 10
f_inspect_end = 2000

ol_d = np.sin(
    2
    * np.pi
    * ((f_chirp_end - f_chirp_start) / t_total * t ** 2 / 2 + f_chirp_start * t)
)

# Plant mechanic characters definition
res_freq = 200
anti_res_freq = 150
res_omega = res_freq * 2 * np.pi
anti_res_omega = anti_res_freq * 2 * np.pi

beta1 = 0.1
beta2 = 0.1

plant_tf = TransferFunc(
    res_omega ** 2
    / anti_res_omega ** 2
    * np.array([1, 2 * beta1 * anti_res_omega, anti_res_omega ** 2]),
    np.array([1, 2 * beta2 * res_omega, res_omega ** 2]),
    dt,
)

# Pvaj initialization
p = np.array([0])
v = np.array([0])
a = np.array([])

# Calculate system response
for i in range(len(ol_d)):
    input_sig = ol_d[i]
    a_current = plant_tf.response(input_sig)  # +random.normal()/4/5
    a = np.append(a, a_current)
    v_current = v[-1] + a_current * (1 / dt)
    v = np.append(v, v_current)
    p_current = p[-1] + v_current * (1 / dt)  # +random.normal()/4/100
    p = np.append(p, p_current)

p = np.delete(p, 0)
v = np.delete(v, 0)
y = np.array(p, dtype=float)
ddy = np.pad(np.diff(y, 2), (2, 0), constant_values=0)

plt.plot(ddy)
