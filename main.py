import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage import io


def laplace_operator(m, n):
    I_m = scipy.sparse.identity(m)
    D_n = scipy.sparse.diags((-2, 1, 1), (0, 1, -1), (n, n))

    I_n = scipy.sparse.identity(n)
    D_m = scipy.sparse.diags((-2, 1, 1), (0, 1, -1), (m, m))

    return scipy.sparse.kron(I_m, D_n) + scipy.sparse.kron(I_n, D_m)


def calculate_C(m, n):
    C = scipy.sparse.eye(m * n, dtype=int, format="lil")
    mask = np.zeros((m * n), dtype=bool)
    mask[m:m * n - m] = True
    C[mask, mask] = 0

    return C


def calculate_F(m, n):
    E = scipy.sparse.eye(m * n, dtype=int, format='lil')
    F = scipy.sparse.eye(m * n, dtype=int, format='lil')
    F -= E

    mask = np.zeros((m * n,), dtype=bool)
    mask[m::m] = True
    mask[2 * m - 1::m] = True
    F[mask, mask] = 1
    mask = np.zeros((m * n), dtype=bool)
    mask[m * n - m:] = True
    F[mask, mask] = 0

    return F


def calculate_H(m, n):
    C = calculate_C(m, n)
    F = calculate_F(m, n)
    H = scipy.sparse.eye(m * n, dtype=int, format='lil')
    H -= C
    H -= F

    return H


def calculate_A(m, n):
    C = calculate_C(m, n)
    F = calculate_F(m, n)
    H = calculate_H(m, n)

    D = laplace_operator(m, n)
    D = H @ D

    A = C + F + D
    return A


def calculate_b(m, n, h, g):
    CF = calculate_C(m, n) + calculate_F(m, n)
    H = calculate_H(m, n)

    b = H @ g.flatten("F") + CF @ h.flatten("F")
    A = calculate_A(m, n)
    return A @ b


def simple_merge(h_star, g, p):
    g = g[1:-1, 1:-1]
    m, n = g.shape
    p = (p[1], p[0])
    h = h_star[p[0] : p[0] + m, p[1] : p[1] + n]

    D = laplace_operator(m, n)
    b = calculate_b(m, n, h, g)
    x, exit_code = scipy.sparse.linalg.cg(D, b, maxiter=10000)

    if exit_code != 0:
        raise ValueError("CG was not successful, exit code is NOT zero")

    if not (p[0] + m <= h_star.shape[0] and p[1] + n <= h_star.shape[1]):
        raise ValueError("Given position is outside of target image")

    h_star[p[0] : p[0] + m, p[1] : p[1] + n] = x.reshape((m, n))
    return h_star


h = io.imread("input/target.jpg")
g = io.imread("input/source.jpg")
p = (5, 5)

red_image = simple_merge(h[:, :, 0].copy(), g[:, :, 0], p)
green_image = simple_merge(h[:, :, 1].copy(), g[:, :, 1], p)
blue_image = simple_merge(h[:, :, 2].copy(), g[:, :, 2], p)

image = np.stack((red_image, green_image, blue_image), axis=2)

print(f"Image shape: {image.shape}")
print(f"Image data type: {image.dtype}")

plt.imshow(image)
plt.show()
