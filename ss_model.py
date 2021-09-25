import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


class StateSpaceModel(object):
    def __init__(self, A, B, C, D, dt):
        n_states = int(np.asarray(A).size ** 0.5)
        self.n_states = n_states
        self.A = np.asarray(A).reshape((n_states, n_states))
        self.B = np.asarray(B).reshape((n_states, -1))
        self.C = np.asarray(C).reshape((-1, n_states))
        self.D = np.asarray(D).reshape((self.C.shape[0], self.B.shape[1]))
        self.dt = dt
        self.x_state = np.zeros((n_states, 1))
        self.y_output = np.zeros((self.C.shape[0], 1))
        self.last_u = np.zeros((self.B.shape[1], 1))

        identity = np.identity(n_states)
        self.expM_zoh_x = expm(A * dt)
        self.expM_zoh_u = np.linalg.inv(A).dot(self.expM_zoh_x - identity).dot(B)

        n_inputs = self.B.shape[1]
        M_linear = np.block(
            [
                [A * dt, B * dt, np.zeros((n_states, n_inputs))],
                [
                    np.zeros((n_inputs, n_states + n_inputs)),
                    np.identity(n_inputs),
                ],
                [np.zeros((n_inputs, n_states + 2 * n_inputs))],
            ]
        )
        expM_linear = expm(M_linear)
        self.Ad_linear = expM_linear[:n_states, :n_states]
        self.Bd1_linear = expM_linear[:n_states, n_states + n_inputs :]
        self.Bd0_linear = (
            expM_linear[:n_states, n_states : n_states + n_inputs] - self.Bd1_linear
        )
        M_step = np.block(
            [[A * dt, B * dt], [np.zeros((n_inputs, n_states + n_inputs))]]
        )
        expM_step = expm(M_step)
        self.Ad_step = expM_step[:n_states, :n_states]
        self.Bd1_step = expM_step[:n_states, n_states:]

    def response(self, u, method="zoh"):
        last_u = self.last_u
        u = np.asarray(u).reshape((-1, 1))
        x_state = self.x_state
        if method == "zoh":  # 教科书般的解法，输入为上一拍的step，延时半拍
            x_state = self.expM_zoh_x.dot(x_state) + self.expM_zoh_u.dot(last_u)
        if method == "zoh2":  # lsim解法，输入为上一拍的step，延时半拍
            x_state = np.dot(self.Ad_step, x_state) + np.dot(self.Bd1_step, last_u)
        if method == "interp":  # 输入为当前拍和上一拍的平均值的step，没有延时
            u_mean = (last_u + u) / 2
            x_state = self.expM_zoh_x.dot(x_state) + self.expM_zoh_u.dot(u_mean)
        elif method == "linear":  # 三角形保持器，forced_response解法，没有延时
            x_state = (
                np.dot(self.Ad_linear, x_state)
                + np.dot(self.Bd0_linear, last_u)
                + np.dot(self.Bd1_linear, u)
            )
        self.last_u = u
        self.x_state = x_state
        self.y_output = self.C.dot(x_state) + self.D.dot(u)
        return self.y_output, x_state


k = 2
m = 0.1
b = 0.0
servo_freq = 10000
dt = 1 / servo_freq
T = 2
t = np.linspace(0, T, servo_freq * T + 1)

A = np.array([[0, 1], [-k / m, -b / m]])
B = np.array([[0], [1 / m]])
C = np.array([1, 0])
D = np.array([[0]])

ss = StateSpaceModel(A, B, C, D, dt)

u = np.linspace(0, 1, len(t))
y = np.zeros_like(u)
for i in range(len(u)):
    input_sig = u[i]
    y_output, x_state = ss.response(input_sig, method="zoh")
    y[i] = y_output
y1 = y * 1

plt.figure()
plt.plot(y1, label="zoh")
plt.legend()
plt.show()
