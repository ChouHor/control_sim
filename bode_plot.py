from scipy import signal
import matplotlib.pyplot as plt

# Q_num, Q_den = signal.butter(3, 2000, "low", analog=True)
#
# sys = signal.TransferFunction(Q_num, Q_den)
sys = signal.TransferFunction([1], [1, 1])
w, mag, phase = signal.bode(sys)

plt.figure()
plt.plot(w, 10 ** (mag / 20))  # Bode magnitude plot
plt.grid()
plt.figure()
plt.plot(w, phase)  # Bode phase plot
plt.grid()
plt.show()

sys = signal.TransferFunction([1], [1, 0])
w, mag, phase = signal.bode(sys)

plt.figure()
plt.plot(w, 10 ** (mag / 20))  # Bode magnitude plot
plt.grid()
plt.figure()
plt.plot(w, phase)  # Bode phase plot
plt.grid()
plt.show()
