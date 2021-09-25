from chirp_data import *


#%% main func implementation


# def main():
#
#     dt = 1 / 600
#
#     t_total = 0.1
#     cross_num = t_total / dt
#     t = np.arange(0, t_total, dt)
#
#     f_chirp_start = 1
#     f_chirp_end = 2000
#
#     f_inspect_start = 10
#     f_inspect_end = 2000
#
#     u = np.sin(
#         2
#         * np.pi
#         * ((f_chirp_end - f_chirp_start) / t_total * t ** 2 / 2 + f_chirp_start * t)
#     )
#
#     res_freq = 300
#     anti_res_freq = 50
#     res_omega = res_freq * 2 * np.pi
#     anti_res_omega = anti_res_freq * 2 * np.pi
#
#     beta1 = 0.1
#     beta2 = 0.1
#
#     plant_pos_tf = TransferFunc(
#         res_omega ** 2
#         / anti_res_omega ** 2
#         * np.array([1, 2 * beta1 * anti_res_omega, anti_res_omega ** 2]),
#         np.array([1, 2 * beta2 * res_omega, res_omega ** 2]),
#         dt,
#     )
#
#     kp = 1
#     ki = 0
#     kd = 0.0001
#     controller_kp = TransferFunc([kp], [1], dt)
#     controller_ki = TransferFunc([ki], [1, 0], dt)
#     controller_kd = TransferFunc([kd, 0], [1], dt)
#     controller_tf = controller_kp + controller_ki + controller_kd
#
#     controller_tf.bode_plot(-1, 2)
#
#     pos = np.array([0])
#     v = np.array([0])
#     a = np.array([])
#
#     for i in range(len(u)):
#         input_sig = u[i]
#         a_current = plant_pos_tf.response(input_sig)  # +random.normal()/4/5
#         a = np.append(a, a_current)
#         v_current = v[-1] + a_current * (1 / dt)
#         v = np.append(v, v_current)
#         p_current = pos[-1] + v_current * (1 / dt)  # +random.normal()/4/100
#         pos = np.append(pos, p_current)
#
#     pos = np.delete(pos, 0)
#     v = np.delete(v, 0)
#     y = np.array(pos, dtype=float)
#
#     y = np.pad(np.diff(y, 2), (2, 0), constant_values=0)
#
#     fig, axes = plt.subplots(1, 2, figsize=(14, 4))
#
#     axes[0].set_xlabel("time / [s]")
#     axes[0].set_ylabel("Force / [N]")
#     axes[0].plot(t, u)
#
#     axes[1].set_xlabel("time / [s]")
#     axes[1].set_ylabel("acceleration / [m/s^2]")
#     axes[1].plot(t, y)
#
#     plt.suptitle("Bode plot")
#     plt.show()
#     print(1)
#
#     u_detrend = u - np.mean(u)
#     y_detrend = y - np.mean(y)
#
#
# if __name__ == "__main__":
#     main()
#%%
