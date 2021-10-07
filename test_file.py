from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from common import *
from control_math import *

from scipy import signal

system = ([1.0], [1.0, 2.0, 1.0])
t, y = signal.impulse(system)
import matplotlib.pyplot as plt

plt.plot(t, y)
plt.show()
