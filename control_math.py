import numpy as np
from sympy.core import symbols
from sympy import simplify, cancel, Poly, fraction, solve
import matplotlib.pyplot as plt
from scipy.linalg import expm


class TransferFunc(object):
    def __init__(self, nom, den, dt):
        self.nom = np.array(nom)
        self.den = np.array(den)
        self.dt = dt
        z = symbols("z")
        _nom_d = _den_d = 0
        # tustin
        s = 2 / dt * (z - 1) / (z + 1)
        # 前向差分
        # s = (z - 1) / dt
        # 后向差分
        # s = (1 - z ** -1) / dt

        for _i in range(len(nom)):
            _nom_d += nom[-_i - 1] * s ** _i
        for _i in range(len(den)):
            _den_d += den[-_i - 1] * s ** _i

        nom_d, den_d = cancel(simplify(_nom_d / _den_d)).as_numer_denom()
        self.nom_d = nom_d.as_poly(z).all_coeffs()
        self.den_d = den_d.as_poly(z).all_coeffs()
        self.input_array = np.zeros_like(self.nom_d, dtype=float)
        self.output_array = np.zeros_like(self.den_d, dtype=float)
        self.output = 0

    def __get_operation_coeffs(self, other):
        s = symbols("s")
        self_nom = Poly(self.nom, s).as_expr(s)
        self_den = Poly(self.den, s).as_expr(s)
        other_nom = Poly(other.nom, s).as_expr(s)
        other_den = Poly(other.den, s).as_expr(s)
        return s, self_nom, self_den, other_nom, other_den

    def __add__(self, other):
        s, self_nom, self_den, other_nom, other_den = self.__get_operation_coeffs(other)
        res_sys_s = cancel(simplify(self_nom / self_den + other_nom / other_den))
        _res_nom, _res_den = res_sys_s.as_numer_denom()
        res_nom = _res_nom.as_poly(s).all_coeffs()
        res_den = _res_den.as_poly(s).all_coeffs()
        res_sys = TransferFunc(res_nom, res_den, self.dt)
        return res_sys

    def __sub__(self, other):
        s, self_nom, self_den, other_nom, other_den = self.__get_operation_coeffs(other)
        res_sys_s = cancel(simplify(self_nom / self_den - other_nom / other_den))
        _res_nom, _res_den = res_sys_s.as_numer_denom()
        res_nom = _res_nom.as_poly(s).all_coeffs()
        res_den = _res_den.as_poly(s).all_coeffs()
        res_sys = TransferFunc(res_nom, res_den, self.dt)
        return res_sys

    def __mul__(self, other):
        s, self_nom, self_den, other_nom, other_den = self.__get_operation_coeffs(other)
        res_sys_s = cancel(simplify(self_nom / self_den * other_nom / other_den))
        _res_nom, _res_den = res_sys_s.as_numer_denom()
        res_nom = _res_nom.as_poly(s).all_coeffs()
        res_den = _res_den.as_poly(s).all_coeffs()
        res_sys = TransferFunc(res_nom, res_den, self.dt)
        return res_sys

    def __truediv__(self, other):
        s, self_nom, self_den, other_nom, other_den = self.__get_operation_coeffs(other)
        res_sys_s = cancel(simplify(self_nom / self_den / (other_nom / other_den)))
        _res_nom, _res_den = res_sys_s.as_numer_denom()
        res_nom = _res_nom.as_poly(s).all_coeffs()
        res_den = _res_den.as_poly(s).all_coeffs()
        res_sys = TransferFunc(res_nom, res_den, self.dt)
        return res_sys

    def zpk(self):
        s = symbols("s")
        self_nom = Poly(self.nom, s).as_expr(s)
        self_den = Poly(self.den, s).as_expr(s)
        z = np.array(solve(self_nom, s))
        p = np.array(solve(self_den, s))
        k = (self_nom / self_den).subs(s, 0)
        return z, p, k

    def response(self, input_sig):
        self.input_array = np.delete(np.insert(self.input_array, 0, input_sig), -1)
        self.output_array = np.delete(np.insert(self.output_array, 0, 0), -1)
        self.output = (
            np.dot(self.input_array, self.nom_d)
            - np.dot(self.output_array[1::], self.den_d[1::])
        ) / self.den_d[0]
        self.output_array[0] = self.output
        return self.output

    def reset(self):
        self.__init__(self.nom, self.den, self.dt)

    def bode(self, f, plot=False):
        # f = np.arange(low, up, 1)
        omega = 2 * np.pi * f
        nom = den = 0
        for i in range(len(self.nom)):
            nom = nom + self.nom[-i - 1] * (1j * omega) ** i
        for i in range(len(self.den)):
            den = den + self.den[-i - 1] * (1j * omega) ** i
        nom = nom.astype("complex")
        den = den.astype("complex")

        fw = nom / den
        gain = 20 * np.log10(np.abs(fw))
        phase = np.angle(fw, deg=True)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            axes[0].set_xlabel("f/[Hz]")
            axes[0].set_ylabel("Gain/[dB]")
            axes[0].semilogx(f, gain)

            axes[1].set_xlabel("f/[Hz]")
            axes[1].set_ylabel("Phase/[deg]")
            axes[1].semilogx(f, phase)

            plt.suptitle("Bode plot")
            plt.show()

        return f, fw


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
        try:
            self.expM_zoh_u = np.linalg.inv(A).dot(self.expM_zoh_x - identity).dot(B)
        except:
            self.expM_zoh_u = None
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
        if method == "zoh":  # 教科书的解法，输入为上一拍的step，延时半拍，可用于模拟带有零阶保持器的系统
            x_state = self.expM_zoh_x.dot(x_state) + self.expM_zoh_u.dot(last_u)
        if method == "zoh2":  # lsim解法，输入为上一拍的step，延时半拍，可用于模拟带有零阶保持器的系统
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


class PID(object):
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_u = np.zeros_like(kp)
        self.ki_output = self.last_u
        self.kd_output = self.last_u

    def response(self, u):
        self.ki_output = self.ki_output + self.ki * u
        self.kd_output = self.kd * (u - self.last_u)
        self.last_u = u
        return self.kp * u + self.ki_output + self.kd_output


def dft(freq, data, dt):
    data = np.array(data)
    t = np.array(range(len(data))) * dt
    cos_wave = np.cos(2 * np.pi * freq * t)
    sine_wave = np.sin(2 * np.pi * freq * t)
    real = np.sum(np.multiply(data, cos_wave))
    imagine = np.sum(np.multiply(data, sine_wave))
    return real + 1j * imagine


def dft_slow(src_data):
    num = len(src_data) - 1
    fw = np.zeros(num, dtype=complex)
    for k in range(num):
        for _n in range(num):
            fw[k] += src_data[_n] * np.e ** (1j * 2 * np.pi / num * _n * k)
    gain = [abs(_) for _ in fw]
    return gain


def dft_vectorized(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    num = x.shape[0]
    n = np.arange(num)
    k = n.reshape((num, 1))
    m = np.exp(-2j * np.pi * k * n / num)
    return np.dot(m, x)


def fft_slow(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    num = x.shape[0]

    if num % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif num <= 32:  # this cutoff should be optimized
        return dft_vectorized(x)
    else:
        x_even = fft_slow(x[::2])
        x_odd = fft_slow(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(num) / num)
        return np.concatenate(
            [
                x_even + factor[: int(num / 2)] * x_odd,
                x_even + factor[int(num / 2) :] * x_odd,
            ]
        )


def _fft(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    num = x.shape[0]

    if np.log2(num) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # _N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    num_min = min(num, 32)

    # Perform an O[N^2] DFT on all length-_N_min sub-problems at once
    n = np.arange(num_min)
    k = n[:, None]
    m = np.exp(-2j * np.pi * n * k / num_min)
    mat_x = np.dot(m, x.reshape((num_min, -1)))

    # build-up each level of the recursive calculation all at once
    while mat_x.shape[0] < num:
        x_even = mat_x[:, : int(mat_x.shape[1] / 2)]
        x_odd = mat_x[:, int(mat_x.shape[1] / 2) :]
        factor = np.exp(-1j * np.pi * np.arange(mat_x.shape[0]) / mat_x.shape[0])[
            :, None
        ]
        mat_x = np.vstack([x_even + factor * x_odd, x_even - factor * x_odd])

    return mat_x.ravel()


def hamm(x):
    x = np.array(x)
    num = len(x)
    hamm_window = 0.53836 - 0.46164 * np.cos(2 * np.pi * np.arange(num) / num)
    x = x * hamm_window

    return x


def anti_hamm(x):
    x = np.array(x)
    num = len(x)
    hamm_window = 0.5 * (1 + np.cos(2 * np.pi * np.arange(num) / (num - 1)))
    x = x * hamm_window

    return x


def half_hamm(x):
    x = np.array(x)
    num = len(x)
    hamm_window = 0.53836 + 0.46164 * np.cos(np.pi * np.arange(num) / (num - 1))

    x = x * hamm_window
    return x


def pad(x):
    num = len(x)
    num_pad = int(2 ** np.ceil(np.log2(num)))
    x_pad = np.pad(x, (0, num_pad - num))
    return x_pad


def fft(x, dt):
    x_pad = pad(x)
    num_pad = len(x_pad)
    fs = 1 / (num_pad * dt)
    f_pad = fs * np.arange(num_pad)

    fw_pad = _fft(x_pad)

    return f_pad, fw_pad


def pole_placement(dB, bandwidth, alpha, servo_freq):
    damping = 0.707
    ratio = 0.5
    k = 10 ** (dB / 20)
    omega = 2 * np.pi * bandwidth
    kp = (1 + 2 * damping * ratio) * omega ** 2 / k
    ki = ratio * omega ** 3 / k / servo_freq
    kd = (2 * damping * omega + ratio * omega - alpha) / k * servo_freq
    return kp, ki, kd


def chirp_iden(sys, start_freq, end_freq, t, plot=False):
    dt = sys.dt
    start_freq_ = 0.8 * start_freq
    end_freq_ = 1.2 * end_freq
    t_ = t * (end_freq_ - start_freq_) / (end_freq - start_freq)
    t_list = np.arange(0, t_, dt)
    pad_len = int(0.1 / dt)

    u = np.sin(
        2
        * np.pi
        * ((end_freq_ - start_freq_) / t_ * t_list ** 2 / 2 + start_freq_ * t_list)
    )
    u = np.pad(u, (0, pad_len))
    a = np.array([])
    for i in range(len(u)):
        input_sig = u[i]
        a_current = sys.response(input_sig)  # +random.normal()/4/5
        a = np.append(a, a_current)

    csv = np.asarray([u, a]).T
    # np.savetxt("datalog.csv", csv, delimiter=",")

    a = np.array(a, dtype=float)
    # a = np.pad(np.diff(pos, 2), (0, 2))
    y = a

    u_detrend = u - np.mean(u)
    y_detrend = y - np.mean(y)

    # f_u, fw_u = fft(half_hamm(u_detrend), dt)
    # f_y, fw_y = fft(half_hamm(y_detrend), dt)
    f_u, fw_u = fft(u_detrend, dt)
    f_y, fw_y = fft(y_detrend, dt)

    resolution = 1 / dt / len(f_u)
    start_point = int(start_freq / resolution)
    end_point = int(end_freq / resolution)

    fw = fw_y / fw_u
    if plot:
        plt.figure(figsize=(14, 4))
        plt.subplot(121)
        plt.xscale("log")
        plt.plot(
            f_u[start_point:end_point],
            20 * np.log10(np.abs(fw)[start_point:end_point]),
        )

        plt.subplot(122)
        plt.xscale("log")
        plt.plot(
            f_u[start_point:end_point],
            np.angle(fw[start_point:end_point], deg=True),
        )
        plt.show()

    return f_u[1:end_point], fw[1:end_point]


def chirp_iden_cross(sys, start_freq, end_freq, t, plot=False):
    dt = sys.dt
    start_freq_ = 0.8 * start_freq
    end_freq_ = 1.2 * end_freq
    t_ = t * (end_freq_ - start_freq_) / (end_freq - start_freq)
    t_list = np.arange(0, t_, dt)
    pad_len = int(0.1 / dt)

    u = np.sin(
        2
        * np.pi
        * ((end_freq_ - start_freq_) / t_ * t_list ** 2 / 2 + start_freq_ * t_list)
    )
    u = np.pad(u, (0, pad_len))
    a = np.array([])
    for i in range(len(u)):
        input_sig = u[i]
        a_current = sys.response(input_sig)  # +random.normal()/4/5
        a = np.append(a, a_current)

    csv = np.asarray([u, a]).T
    # np.savetxt("datalog.csv", csv, delimiter=",")

    a = np.array(a, dtype=float)
    y = a

    u_detrend = u - np.mean(u)
    y_detrend = y - np.mean(y)

    cross_num = int(len(u))
    Ruy = np.zeros(cross_num)
    Ruu = np.zeros(cross_num)
    for i in range(cross_num):
        print(i)
        Ruy[i] = sum(u_detrend[0 : -i - 1] * y_detrend[i:-1])
        Ruu[i] = sum(u_detrend[0 : -i - 1] * u_detrend[i:-1])
    # Ruy = half_hamm(Ruy)
    # Ruu = half_hamm(Ruu)
    f_u, fw_Ruy = fft(Ruy, dt)
    f_y, fw_Ruu = fft(Ruu, dt)
    fw = fw_Ruy / fw_Ruu

    resolution = 1 / dt / len(f_u)
    start_point = int(start_freq / resolution)
    end_point = int(end_freq / resolution)

    if plot:
        plt.figure(figsize=(14, 4))
        plt.subplot(121)
        plt.xscale("log")
        plt.plot(
            f_u[start_point:end_point],
            20 * np.log10(np.abs(fw)[start_point:end_point]),
        )

        plt.subplot(122)
        plt.xscale("log")
        plt.plot(
            f_u[start_point:end_point],
            np.angle(fw[start_point:end_point], deg=True),
        )
        plt.show()

    return f_u[1:end_point], fw[1:end_point]


def chirp_iden_pos(sys, start_freq, end_freq, t, plot=False):
    dt = sys.dt
    start_freq_ = 0.8 * start_freq
    end_freq_ = 1.2 * end_freq
    t_ = t * (end_freq_ - start_freq_) / (end_freq - start_freq)
    t_list = np.arange(0, t_, dt)
    pad_len = int(0.1 / dt)

    u = np.sin(
        2
        * np.pi
        * ((end_freq_ - start_freq_) / t * t_list ** 2 / 2 + start_freq_ * t_list)
    )
    u = np.pad(u, (0, pad_len))
    a = np.array([])
    v = np.array([0])
    p = np.array([0])
    for i in range(len(u)):
        input_sig = u[i]
        a_current = sys.response(input_sig)  # +random.normal()/4/5
        v_current = v[-1] + a_current * dt
        p_current = p[-1] + v_current * dt
        a = np.append(a, a_current)
        v = np.append(v, v_current)
        p = np.append(p, p_current)

    a = np.array(a, dtype=float)
    v = np.array(v, dtype=float)
    p = np.array(p[1:], dtype=float)
    # a = np.pad(np.diff(pos, 2), (0, 2))
    y = np.diff(p, 2) / dt / dt
    y = np.pad(y, (1, 1), "constant", constant_values=(0, 0))

    u_detrend = u - np.mean(u)
    y_detrend = y - np.mean(y)

    # f_u, fw_u = fft(half_hamm(u_detrend), dt)
    # f_y, fw_y = fft(half_hamm(y_detrend), dt)
    f_u, fw_u = fft(u_detrend, dt)
    f_y, fw_y = fft(y_detrend, dt)

    resolution = 1 / dt / len(f_u)
    start_point = int(start_freq / resolution)
    end_point = int(end_freq / resolution)

    fw = fw_y / fw_u
    if plot:
        plt.figure(figsize=(14, 4))
        plt.subplot(121)
        plt.xscale("log")
        plt.plot(
            f_u[start_point:end_point],
            20 * np.log10(np.abs(fw)[start_point:end_point]),
        )

        plt.subplot(122)
        plt.xscale("log")
        plt.plot(
            f_u[start_point:end_point],
            np.angle(fw[start_point:end_point], deg=True),
        )
        plt.show()

    return f_u[1:end_point], fw[1:end_point]
