import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage import io


def laplace_operator(n, m):
    I_m = scipy.sparse.identity(m)
    D_n = scipy.sparse.diags((-2, 1, 1), (0, 1, -1), (n, n))

    I_n = scipy.sparse.identity(n)
    D_m = scipy.sparse.diags((-2, 1, 1), (0, 1, -1), (m, m))

    return scipy.sparse.kron(I_m, D_n) + scipy.sparse.kron(I_n, D_m)


def simple_merge(h, g, p):
    g = g[1:-1, 1:-1]
    p = (p[1], p[0])
    n, m = g.shape

    D = laplace_operator(n, m)
    b = D @ g.flatten("C")
    x, exit_code = scipy.sparse.linalg.cg(D, b, maxiter=10000)

    if exit_code != 0:
        raise ValueError("CG was not successful, exit code is NOT zero")

    if not (p[0] + n <= h.shape[0] and p[1] + m <= h.shape[1]):
        raise ValueError("Given position is outside of target image")

    h[p[0] : p[0] + n, p[1] : p[1] + m] = x.reshape((n, m))
    return h


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
