import numpy as np
from sympy.core import symbols
from sympy import simplify, cancel, Poly, fraction
from scipy.linalg import hankel
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt


class TransferFunc(object):
    def __init__(self, nom, den, dt):
        self.nom = nom
        self.den = den
        self.dt = dt
        z = symbols("z")
        _nom_d = _den_d = 0
        for _i in range(len(nom)):
            _nom_d += nom[-_i - 1] * (2 / dt * (z - 1) / (z + 1)) ** _i
        for _i in range(len(den)):
            _den_d += den[-_i - 1] * (2 / dt * (z - 1) / (z + 1)) ** _i
        nom_d, den_d = cancel(simplify(_nom_d / _den_d)).as_numer_denom()
        self.nom_d = nom_d.as_poly(z).all_coeffs()
        self.den_d = den_d.as_poly(z).all_coeffs()
        self.input_array = np.zeros_like(self.nom_d)
        self.output_array = np.zeros_like(self.den_d)
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

    def response(self, input_sig):
        self.input_array = np.delete(np.insert(self.input_array, 0, input_sig), -1)
        self.output_array = np.delete(np.insert(self.output_array, 0, 0), -1)
        self.output = (
            np.dot(self.input_array, self.nom_d)
            - np.dot(self.output_array[1::], self.den_d[1::])
        ) / self.den_d[0]
        self.output_array[0] = self.output
        return self.output

    def bode_plot(self, low, up):
        # f = 10 ** np.arange(low, up, 0.01)
        f = np.arange(low, up, 1)
        omega = 2 * np.pi * f
        nom = den = 0
        for i in range(len(self.nom)):
            nom = nom + self.nom[-i - 1] * (1j * omega) ** i
        for i in range(len(self.den)):
            den = den + self.den[-i - 1] * (1j * omega) ** i
        nom = nom.astype("complex")
        den = den.astype("complex")

        gain = 20 * np.log10(np.abs(nom / den))
        phase = np.angle(nom / den, deg=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].set_xlabel("f/[Hz]")
        axes[0].set_ylabel("Gain/[dB]")
        axes[0].semilogx(f, gain)

        axes[1].set_xlabel("f/[Hz]")
        axes[1].set_ylabel("Phase/[deg]")
        axes[1].semilogx(f, phase)

        plt.suptitle("Bode plot")
        plt.show()


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


def chirp_iden(sys, start_freq, end_freq, t):
    dt = sys.dt
    t_list = np.arange(0, t, dt)
    u = np.sin(
        2
        * np.pi
        * ((end_freq - start_freq) / t * t_list ** 2 / 2 + start_freq * t_list)
    )
    a = np.array([])
    for i in range(len(u)):
        input_sig = u[i]
        a_current = sys.response(input_sig)  # +random.normal()/4/5
        a = np.append(a, a_current)

    a = np.array(a, dtype=float)
    # a = np.pad(np.diff(p, 2), (0, 2))
    y = a

    u_detrend = u - np.mean(u)
    y_detrend = y - np.mean(y)

    f_u, fw_u = fft(half_hamm(u_detrend), dt)
    f_y, fw_y = fft(half_hamm(y_detrend), dt)

    resolution = 1 / dt / len(f_u)
    start_point = int(start_freq / resolution)
    end_point = int(end_freq / resolution)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.xscale("log")
    plt.plot(
        f_u[start_point:end_point],
        20
        * np.log10(
            np.abs(fw_y)[start_point:end_point] / np.abs(fw_u)[start_point:end_point]
        ),
    )

    plt.subplot(122)
    plt.xscale("log")
    plt.plot(
        f_u[start_point:end_point],
        20 * np.angle((fw_y)[start_point:end_point] / (fw_u)[start_point:end_point]),
    )
    plt.show()

    # plt.figure(1)
    # plt.plot(t_list, u)
    # plt.show()
    # plt.figure(2)
    # plt.plot(t_list, a)
    # plt.show()
