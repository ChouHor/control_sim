from scipy.linalg import hankel, toeplitz
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


b, a = signal.butter(6, 100, "low", analog=True)
w, h = signal.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title("Butterworth filter frequency response")
plt.xlabel("Frequency [radians / second]")
plt.ylabel("Amplitude [dB]")
plt.margins(0, 0.1)
plt.grid(which="both", axis="both")
plt.axvline(100, color="green")  # cutoff frequency
plt.show()

sys = signal.lti(b, a)

w, mag, phase = sys.bode()

plt.figure()
plt.semilogx(w, mag)  # Bode magnitude plot
plt.figure()
plt.semilogx(w, phase)  # Bode phase plot
plt.show()
