import numpy as np
import scipy


def laplace_operator(m, n):
    I_m = scipy.sparse.identity(m)
    D_n = scipy.sparse.diags((-2, 1, 1), (0, 1, -1), (n, n))

    I_n = scipy.sparse.identity(n)
    D_m = scipy.sparse.diags((-2, 1, 1), (0, 1, -1), (m, m))

    return scipy.sparse.kron(I_m, D_n) + scipy.sparse.kron(I_n, D_m)


m = 3
n = 6

C = scipy.sparse.eye(m * n, dtype=int, format="lil")
mask = np.zeros((m * n), dtype=bool)
mask[m : m * n - m] = True
C[mask, mask] = 0
# print(C.toarray())


E = scipy.sparse.eye(m * n, dtype=int, format="lil")

F = scipy.sparse.eye(m * n, dtype=int, format="lil")
F -= E

mask = np.zeros((m * n,), dtype=bool)
mask[m::m] = True
mask[2 * m - 1 :: m] = True
F[mask, mask] = 1
mask = np.zeros((m * n), dtype=bool)
mask[m * n - m :] = True
F[mask, mask] = 0

H = scipy.sparse.eye(m * n, dtype=int, format="lil")
H -= C
H -= F

D = laplace_operator(m, n)
D = H @ D

R = C + F + D
print(R.toarray())
