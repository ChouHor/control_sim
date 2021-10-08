import numpy as np

# from numpy import sin, cos
from sympy import symbols, sin, cos


def cross(x, y):
    return np.cross(x.T4chirp, y.T4chirp).T


def forward(Tip1i, PCip1ip1, Iip1Cip1, mip1, dthetaip1, ddthetaip1, wii, dwii, dvii):
    Riip1 = Tip1i[0:3, 0:3].T4chirp
    Pip1i = Tip1i[0:3, 3]
    wip1ip1 = Riip1.dot(wii) + dthetaip1 * Z
    dwip1ip1 = Riip1.dot(dwii) + cross(Riip1.dot(wii), dthetaip1 * Z) + ddthetaip1 * Z
    dvip1ip1 = Riip1.dot(cross(dwii, Pip1i) + cross(wii, cross(wii, Pip1i)) + dvii)
    dvCip1ip1 = (
        cross(dwip1ip1, PCip1ip1) + cross(wip1ip1, cross(wip1ip1, PCip1ip1)) + dvip1ip1
    )
    Fip1ip1 = mip1 * dvCip1ip1
    Nip1ip1 = Iip1Cip1.dot(dwip1ip1) + cross(wip1ip1, Iip1Cip1.dot(wip1ip1))
    return wip1ip1, dwip1ip1, dvip1ip1, Fip1ip1, Nip1ip1


def backward(Tip1i, PCii, Fii, Nii, fip1ip1, nip1ip1):
    Rip1i = Tip1i[0:3, 0:3]
    Pip1i = Tip1i[0:3, 3]
    fii = Rip1i.dot(fip1ip1) + Fii
    nii = Nii + Rip1i.dot(nip1ip1) + cross(PCii, Fii) + cross(Pip1i, Rip1i.fip1ip1)
    taui = nii.T4chirp.dot(Z)
    return fii, nii, taui


theta1 = symbols("theta1")
dtheta1 = symbols("dtheta1")
ddtheta1 = symbols("ddtheta1")
theta2 = symbols("theta2")
dtheta2 = symbols("dtheta2")
ddtheta2 = symbols("ddtheta2")
theta3 = 0
dtheta3 = 0
ddtheta3 = 0
l1 = 1
l2 = 1
T10 = np.array(
    [
        [cos(theta1), -sin(theta1), 0, 0],
        [sin(theta1), cos(theta1), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
T21 = np.array(
    [
        [cos(theta2), -sin(theta2), 0, l1],
        [sin(theta2), cos(theta2), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
T32 = np.array([[1, 0, 0, l2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
X = np.array([[1], [0], [0]])
Y = np.array([[0], [1], [0]])
Z = np.array([[0], [0], [1]])
I0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
vec0 = np.array([[0], [0], [0]])


m1 = 1
m2 = 1
m3 = 1
g = 9.8
PC11 = np.array([[l1 / 2], [0], [0]])
PC22 = np.array([[l2 / 2], [0], [0]])
PC33 = np.array([[0], [0], [0]])
# I1C1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# I2C2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# I3C3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
I1C1 = np.random.random([3, 3])
I2C2 = np.random.random([3, 3])
I3C3 = np.random.random([3, 3])
w00 = vec0
dw00 = vec0
dv00 = vec0
# dv00 = g * Y
f33 = vec0
n33 = vec0


w11, dw11, dv11, F11, N11 = forward(
    T10, PC11, I1C1, m1, dtheta1, ddtheta1, w00, dw00, dv00
)
print(w11, dw11, dv11, F11, N11)
w22, dw22, dv22, F22, N22 = forward(
    T21, PC22, I2C2, m2, dtheta2, ddtheta2, w11, dw11, dv11
)
print(w22, dw22, dv22, F22, N22)
w33, dw33, dv33, F33, N33 = forward(
    T32, PC33, I3C3, m3, dtheta3, ddtheta3, w22, dw22, dv22
)
print(w33, dw33, dv33, F33, N33)
