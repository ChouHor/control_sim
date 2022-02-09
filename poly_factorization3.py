import numpy as np

# a**2+b**2+a**2*b+a*b**2+a*b+2*a+b+2
# 1/2*[a b]*[2+2*b       2*a+2*b+1]*[a] + [a b][2] + 2 = 0
#           [2*a+2*b+1   2+2*a    ] [b]        [1]
# A = [2 1]
#     [1 2]
# B = [2]
#     [1]
a = 1
b = 1
A = np.array([[2 + 2 * b, 2 * a + 2 * b + 1], [2 * a + 2 * b + 1, 2 + 2 * a]])
B = -np.array([[2], [1]])
print(np.linalg.pinv(A) @ B)
