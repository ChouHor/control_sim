from chirp_data import *

#%% Plot force input and acc response

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].set_xlabel("time / [s]")
axes[0].set_ylabel("Force / [N]")
axes[0].plot(t, ol_d)

axes[1].set_xlabel("time / [s]")
axes[1].set_ylabel("acc / [m/s^2]")
axes[1].plot(t, ddy)

plt.suptitle("Force input and acc response")
plt.show()

plant_tf.bode_plot(0, 3)
# controller_tf.bode_plot(0, 3)
# closed_loop_tf.bode_plot(0, 3)
