from scipy import signal
import matplotlib.pyplot as plt

Q_num, Q_den = signal.butter(3, 2000, "low", analog=True)

sys = signal.TransferFunction(Q_num, Q_den)
w, mag, phase = signal.bode(sys)

plt.figure()
plt.semilogx(w, mag)  # Bode magnitude plot
plt.figure()
plt.semilogx(w, phase)  # Bode phase plot
plt.show()
