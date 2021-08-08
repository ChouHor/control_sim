from common import *
import numpy as np
from control_math import *
import matplotlib.pyplot as plt
from itertools import chain, zip_longest

res_freq = 200
anti_res_freq = 150
res_omega = res_freq * 2 * np.pi
anti_res_omega = anti_res_freq * 2 * np.pi

beta1 = 0.1
beta2 = 0.1

sys = TransferFunc(
    res_omega ** 2
    / anti_res_omega ** 2
    * np.array([1, 2 * beta1 * anti_res_omega, anti_res_omega ** 2]),
    np.array([1, 2 * beta2 * res_omega, res_omega ** 2]),
    DT,
)
identity_tf = TransferFunc([1], [1], DT)
# sys = TransferFunc([1], [1, 2, 1], DT)
# sys = TransferFunc([1, 2, 1], [1, 3, 5], DT)
# sys = identity_tf / sys

f, fw = sys.bode(np.array(range(10, 5000)), plot=False)


# def fit(f, fw, bn, am):
#     s = 1j * 2 * np.pi * f
#     num = len(f)
#     A = np.zeros([num, am], dtype="complex")
#     for i in range(am):
#         A[:, i] = s ** i * fw
#     Re_A = A.real
#     Im_A = A.imag
#     for i in range(bn + 1):
#         if i % 2 == 0:
#             Re_A = np.concatenate((Re_A, -(s ** i).real.reshape(num, 1)), axis=1)
#         else:
#             Im_A = np.concatenate((Im_A, -(s ** i).imag.reshape(num, 1)), axis=1)
#
#     Re_A = np.mat(Re_A)[:, 1:]
#     Im_A = np.mat(Im_A)[:, 1:]
#     B = np.mat(-fw * s ** am)
#     Re_B = np.mat(B.real).T
#     Im_B = np.mat(B.imag).T
#     Re_X = ((Re_A.T * Re_A).I * Re_A.T * Re_B).real.tolist()  # 最小二乘法
#     Im_X = ((Im_A.T * Im_A).I * Im_A.T * Im_B).real.tolist()  # 最小二乘法
#
#     # Re_X = (np.linalg.pinv(Re_A) * Re_B).real.tolist()  # numpy库求伪逆
#     # Im_X = (np.linalg.pinv(Im_A) * Re_B).real.tolist()  # numpy库求伪逆
#
#     Re_X = [i[0] for i in Re_X]
#     Im_X = [i[0] for i in Im_X]
#     Bn1 = Re_X[am - 1 :]  # [b0, b2, b4, ...]
#     Bn2 = Im_X[am - 1 :]  # [b1, b3, b5, ...]
#     print(Bn1, Bn2)
#
#     Bn = [x for x in chain.from_iterable(zip_longest(Bn1, Bn2)) if x is not None]
#     Am = [0] + Re_X[0 : am - 1] + [1]
#     Bn.reverse()
#     Am.reverse()
#     return Bn, Am


def fit(f, fw, bn, am):
    s = 1j * 2 * np.pi * f
    num = len(f)
    A = np.zeros([num, am], dtype="complex")
    for i in range(am):
        A[:, i] = s ** i * fw
    Re_A = A.real
    Im_A = A.imag
    for i in range(bn + 1):
        if i % 2 == 0:
            Re_A = np.concatenate((Re_A, -(s ** i).real.reshape(num, 1)), axis=1)
        else:
            Im_A = np.concatenate((Im_A, -(s ** i).imag.reshape(num, 1)), axis=1)

    Re_A = np.mat(Re_A)
    Im_A = np.mat(Im_A)
    B = np.mat(-fw * s ** am)
    Re_B = np.mat(B.real).T
    Im_B = np.mat(B.imag).T
    Re_X = ((Re_A.T * Re_A).I * Re_A.T * Re_B).real.tolist()  # 最小二乘法
    Im_X = ((Im_A.T * Im_A).I * Im_A.T * Im_B).real.tolist()  # 最小二乘法

    # Re_X = (np.linalg.pinv(Re_A) * Re_B).real.tolist()  # numpy库求伪逆
    # Im_X = (np.linalg.pinv(Im_A) * Re_B).real.tolist()  # numpy库求伪逆

    Re_X = [i[0] for i in Re_X]
    Im_X = [i[0] for i in Im_X]
    Bn1 = Re_X[am:]  # [b0, b2, b4, ...]
    Bn2 = Im_X[am:]  # [b1, b3, b5, ...]

    Bn = [x for x in chain.from_iterable(zip_longest(Bn1, Bn2)) if x is not None]
    Am = Re_X[0:am] + [1]
    Bn.reverse()
    Am.reverse()
    return Bn, Am


Bn, Am = fit(f, fw, 2, 2)

print(sys.nom, sys.den)
print(Bn, Am)

fit_sys = TransferFunc(Bn, Am, DT)
f_fit, fw_fit = fit_sys.bode(np.array(range(10, 5000)))

gain = 20 * np.log10(np.abs(fw))
phase = np.angle(fw, deg=True)
gain_fit = 20 * np.log10(np.abs(fw_fit))
phase_fit = np.angle(fw_fit, deg=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0][0].set_xlabel("f/[Hz]")
axes[0][0].set_ylabel("Gain/[dB]")
axes[0][0].semilogx(f, gain)

axes[0][1].set_xlabel("f/[Hz]")
axes[0][1].set_ylabel("Phase/[deg]")
axes[0][1].semilogx(f, phase)

axes[1][0].set_xlabel("f/[Hz]")
axes[1][0].set_ylabel("Gain/[dB]")
axes[1][0].semilogx(f_fit, gain_fit)

axes[1][1].set_xlabel("f/[Hz]")
axes[1][1].set_ylabel("Phase/[deg]")
axes[1][1].semilogx(f_fit, phase_fit)

plt.suptitle("Bode plot")
plt.show()
