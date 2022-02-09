import numpy as np

# a=3, b=4, 24
# (a+1)*(b+2)=ab+2a+b+2
# J = [b+2 a+1]
#
# 1/2*[a b]*[0 1]*[a] + [a b][2] + 2 = 0
#           [1 0] [b]        [1]
# A = [0 2]
#     [2 0]
# B = [2]
#     [1]
a = 3.1
b = 4.1
sol = np.array([a, b]).reshape(-1, 1)
for i in range(100):
    a = sol[0, 0]
    b = sol[1, 0]
    J = np.array([b + 2, a + 1]).reshape(-1, 1)
    fk = a * b + 2 * a + b + 2 - 24
    delta_sol = np.linalg.pinv(J) * fk
    sol = sol - delta_sol
