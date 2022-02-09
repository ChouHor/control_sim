import numpy as np
import matplotlib.pyplot as plt

# a=3, b=4, 24
# (a+1)*(b+2)=a*b+2*a+b+2  24
# (a+2)*(b+3)=a*b+3*a+2*b+6  35
# (a+3)*(b+4)=a*b+4*a+3*b+12  48
# J = [b+2 a+1]
#     [b+3 a+2]
#     [b+4 a+3]
#
# 1/2*[a b]*[0 1]*[a] + [a b][2] + 2 = 0
#           [1 0] [b]        [1]
# A = [0 2]
#     [2 0]
# B = [2]
#     [1]
a = 3.1
b = 5.1
sol = np.array([a, b]).reshape(-1, 1)
num = 1000
sol_H_list = np.zeros((2, num))

for i in range(num):
    a = sol[0, 0]
    b = sol[1, 0]
    J = np.array([[b + 2, a + 1], [b + 3, a + 2], [b + 4, a + 3]]).reshape(-1, 2)
    f1 = a * b + 2 * a + b + 2 - 24.1
    f2 = a * b + 3 * a + 2 * b + 6 - 34.9
    f3 = a * b + 4 * a + 3 * b + 12 - 48.5
    fk = np.array([f1, f2, f3]).reshape(-1, 1)
    weight = np.diag([1, 1, 1])
    delta_sol = np.linalg.pinv(J) @ weight @ fk
    sol = sol - 0.01 * delta_sol
    sol_H_list[:, i] = delta_sol.reshape(-1)

print((a + 1) * (b + 2))
print((a + 2) * (b + 3))
print((a + 3) * (b + 4))

plt.figure()
plt.plot(sol_H_list[0], label="0")
plt.figure()
plt.plot(sol_H_list[1], label="1")
plt.legend()
plt.show()
