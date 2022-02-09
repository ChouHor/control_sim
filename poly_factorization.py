import matplotlib.pyplot as plt
import numpy as np

# a**2+2*b**2+0.3*a*b+4*a+5*b+6
# 1/2*[a b]*[2 3]*[a] + [a b][4] + 6 = 0
#           [3 4] [b]        [5]
# J = [2*a+3*b+4 4*b+3*a+5]
# A = [2 1]
#     [1 2]
# B = [4]
#     [5]
A = np.array([[2, 0.3], [0.3, 4]])
B = -np.array([[4], [5]])
sol = np.linalg.pinv(A) @ B
print(sol)
a = sol[0, 0]
b = sol[1, 0]
print(a ** 2 + 2 * b ** 2 + 0.3 * a * b + 4 * a + 5 * b + 6)
a = 1
b = 1
sol_H = np.array([a, b]).reshape(-1, 1)
H = A
num = 100
sol_H_list = np.zeros((2, num))
sol_Ax_list = np.zeros((2, num))
for i in range(num):
    a = sol[0, 0]
    b = sol[1, 0]
    J = np.array([2 * a + 0.3 * b + 4, 4 * b + 0.3 * a + 5]).reshape(-1, 1)
    delta_sol_H = np.linalg.pinv(H) @ J
    sol_H = sol_H - 0.1 * delta_sol_H
    sol_H_list[:, i] = delta_sol_H.reshape(-1)
print(a ** 2 + 2 * b ** 2 + 0.3 * a * b + 4 * a + 5 * b + 6)

a = 1
b = 1
sol_Ax = np.array([a, b]).reshape(-1, 1)
for i in range(num):
    a = sol_Ax[0, 0]
    b = sol_Ax[1, 0]
    J = np.array([2 * a + 0.3 * b + 4, 4 * b + 0.3 * a + 5]).reshape(-1, 1)
    delta_sol_Ax = A @ sol_Ax - B
    sol_Ax = sol_Ax - 0.1 * delta_sol_Ax
    sol_Ax_list[:, i] = delta_sol_Ax.reshape(-1)

print(a ** 2 + 2 * b ** 2 + 0.3 * a * b + 4 * a + 5 * b + 6)
# plt.figure()
# plt.plot(sol_H_list[0], label="sol_H_list0")
# plt.plot(sol_Ax_list[0], label="sol_Ax_list")
# plt.legend()
# plt.figure()
# plt.plot(sol_H_list[1], label="sol_H_list1")
# plt.plot(sol_Ax_list[1], label="sol_Ax_list1")
# plt.legend()
# plt.show()
