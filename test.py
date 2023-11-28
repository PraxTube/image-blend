import numpy as np
import scipy


def laplace_operator(n, m):
    I_m = scipy.sparse.identity(m)
    D_n = scipy.sparse.diags((-2, 1, 1), (0, 1, -1), (n, n))

    I_n = scipy.sparse.identity(n)
    D_m = scipy.sparse.diags((-2, 1, 1), (0, 1, -1), (m, m))

    return scipy.sparse.kron(I_m, D_n) + scipy.sparse.kron(I_n, D_m)


n = 4  # Change this value to adjust the size of the array

C = scipy.sparse.eye(n * n, dtype=int, format="lil")
mask = np.zeros((n * n), dtype=bool)
mask[n:n * n - n] = True
C[mask, mask] = 0
# print(C.toarray())


E = scipy.sparse.eye(n * n, dtype=int, format='lil')

F = scipy.sparse.eye(n * n, dtype=int, format='lil')
F -= E

mask = np.zeros((n * n,), dtype=bool)
mask[n::n] = True
mask[2 * n - 1::n] = True
F[mask, mask] = 1
mask = np.zeros((n * n), dtype=bool)
mask[n * n - n:] = True
F[mask, mask] = 0

H = scipy.sparse.eye(n * n, dtype=int, format='lil')
H -= C
H -= F

D = laplace_operator(n, n)
D = H @ D

R = C + F + D
print(R.toarray())
