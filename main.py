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
    mask[m : m * n - m] = True
    C[mask, mask] = 0

    return C


def calculate_F(m, n):
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

    return F


def calculate_H(m, n):
    C = calculate_C(m, n)
    F = calculate_F(m, n)
    H = scipy.sparse.eye(m * n, dtype=int, format="lil")
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


def naive_blending(h_star, g, p):
    m, n = g.shape
    p = (p[1], p[0])
    h_star[p[0] : p[0] + m, p[1] : p[1] + n] = g
    return h_star


def laplace_blending(h_star, g, p):
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


def gray_laplace_blending(h, g, p):
    h_gray = (h * 255).astype(int)
    g_gray = (g * 255).astype(int)

    image = laplace_blending(h_gray, g_gray, p).astype(float) / 255.0
    return image


def bear_water_example(naive_blend=False):
    h = io.imread("input/water.jpg")
    g = io.imread("input/bear.jpg")
    p = (10, 80)

    if naive_blend:
        red_image = naive_blending(h[:, :, 0].copy(), g[:, :, 0], p)
        green_image = naive_blending(h[:, :, 1].copy(), g[:, :, 1], p)
        blue_image = naive_blending(h[:, :, 2].copy(), g[:, :, 2], p)
    else:
        red_image = laplace_blending(h[:, :, 0].copy(), g[:, :, 0], p)
        green_image = laplace_blending(h[:, :, 1].copy(), g[:, :, 1], p)
        blue_image = laplace_blending(h[:, :, 2].copy(), g[:, :, 2], p)

    image = np.stack((red_image, green_image, blue_image), axis=2)
    return image


def plane_bird_example(naive_blend=False):
    h = io.imread("input/bird.jpg")
    g = io.imread("input/plane.jpg")
    p = (400, 50)

    if naive_blend:
        red_image = naive_blending(h[:, :, 0].copy(), g[:, :, 0], p)
        green_image = naive_blending(h[:, :, 1].copy(), g[:, :, 1], p)
        blue_image = naive_blending(h[:, :, 2].copy(), g[:, :, 2], p)
    else:
        red_image = laplace_blending(h[:, :, 0].copy(), g[:, :, 0], p)
        green_image = laplace_blending(h[:, :, 1].copy(), g[:, :, 1], p)
        blue_image = laplace_blending(h[:, :, 2].copy(), g[:, :, 2], p)

    image = np.stack((red_image, green_image, blue_image), axis=2)
    return image


def main():
    print("WARNING: This will take some time. The program will compute a lot of redundant matrices.")
    images = [
        bear_water_example(True),
        plane_bird_example(True),
        bear_water_example(),
        plane_bird_example(),
    ]
    titles = [
        "Naive Blend - Bear in Water",
        "Naive Blend - Plane and Bird",
        "Laplace Blend - Bear in Water",
        "Laplace Blend - Plane and Bird",
    ]

    # Create a figure and subplots based on the number of images
    fig, axes = plt.subplots(2, 2, figsize=(2 * 5, 2 * 5))

    for i in range(len(images)):
        row = i // 2
        col = i % 2
        axes[row, col].imshow(images[i])
        axes[row, col].set_title(titles[i])
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # main()
    print(laplace_operator(5, 7).toarray())
