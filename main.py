"""
Script that blends a source image into a bigger target image.
It only works for rectangular images, well it _should_ but
there are some bugs so it doesn't really work.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage import io


def laplace_operator(m, n):
    """
    Compute laplace operator given m rows and n columns.
    """
    I_m = scipy.sparse.identity(m)
    D_n = scipy.sparse.diags((-2, 1, 1), (0, 1, -1), (n, n))

    I_n = scipy.sparse.identity(n)
    D_m = scipy.sparse.diags((-2, 1, 1), (0, 1, -1), (m, m))

    return scipy.sparse.kron(I_m, D_n) + scipy.sparse.kron(I_n, D_m)


def discrete_gradient(f):
    """
    Compute the discrete gradient of the given n x m matrix
    """
    n, m = f.shape
    D_n = scipy.sparse.diags((1, -1), (0, 1), (n, n))
    D_m = scipy.sparse.diags((1, -1), (0, 1), (m, m))

    diff_x = D_n @ f
    diff_y = f @ D_m.T

    gradient = np.stack((diff_x, diff_y), axis=2)
    return gradient


def calculate_C(m, n):
    """
    Compute the outer 1's of the Laplace Matrix.
    These 1's represent the columns of the edges of Omega.
    """
    C = scipy.sparse.eye(m * n, dtype=int, format="lil")
    mask = np.zeros((m * n), dtype=bool)
    mask[m : m * n - m] = True
    C[mask, mask] = 0
    return C


def calculate_F(m, n):
    """
    Compute the inner 1's of the Laplace Matrix.
    These 1's represent the rows of the edges of Omega.
    """
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
    """
    Compute the 1's which will mask the actual edge values of h.
    """
    C = calculate_C(m, n)
    F = calculate_F(m, n)
    H = scipy.sparse.eye(m * n, dtype=int, format="lil")
    H -= C
    H -= F

    return H


def calculate_A(m, n):
    """
    Compute the Laplace Matrix that takes into account the edges of Omega.
    """
    C = calculate_C(m, n)
    F = calculate_F(m, n)
    H = calculate_H(m, n)

    D = laplace_operator(m, n)
    D = H @ D

    A = C + F + D
    return A


def calculate_b(m, n, h, g):
    """
    Compute b, where b takes into account the edges of Omega.
    B is the right hand side of our CG equation.
    """
    CF = calculate_C(m, n) + calculate_F(m, n)
    H = calculate_H(m, n)

    b = H @ g.flatten("F") + CF @ h.flatten("F")
    A = calculate_A(m, n)
    return A @ b


def calculate_div_v(n, m, h, g):
    """
    Calculate the divergenz of the vector field.
    It takes into account the magnitude of both gradients.
    """
    div_h = discrete_gradient(h)
    div_g = discrete_gradient(g)

    mag_div_h = np.linalg.norm(div_h, axis=2)
    mag_div_g = np.linalg.norm(div_g, axis=2)

    mask = mag_div_h > mag_div_g
    inv_mask = mag_div_h <= mag_div_g

    mask = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    inv_mask = np.repeat(inv_mask[:, :, np.newaxis], 2, axis=2)

    v = div_h * mask + div_g * inv_mask

    D_n = scipy.sparse.diags((1, -1), (0, -1), (n, n))
    D_m = scipy.sparse.diags((1, -1), (0, -1), (m, m))

    r = D_n @ v[:, :, 0] + v[:, :, 1] @ D_m.T
    return r


def naive_blending(h_star, g, p):
    """
    Copy paste the source into the target.
    """
    m, n = g.shape
    p = (p[1], p[0])
    h_star[p[0] : p[0] + m, p[1] : p[1] + n] = g
    return h_star


def laplace_blending(h_star, g, p):
    """
    Use laplace blending to blend the source into the target.
    """
    # The inner of Omega only. The edges will be set later.
    g = g[1:-1, 1:-1]
    m, n = g.shape
    # Swap values as we need row first, column second (y, x)
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


def mix_blend(h_star, g, p):
    # The inner of Omega only. The edges will be set later.
    g = g[1:-1, 1:-1]
    m, n = g.shape
    # Swap values as we need row first, column second (y, x)
    p = (p[1], p[0])
    h = h_star[p[0] : p[0] + m, p[1] : p[1] + n]

    D = laplace_operator(m, n)
    b = calculate_div_v(m, n, h, g).flatten()
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


def bear_water_example(blend_type):
    # Read in the images and set pixel offset `p`
    h = io.imread("input/water.jpg")
    g = io.imread("input/bear.jpg")
    p = (10, 80)

    if blend_type == 0:
        red_image = naive_blending(h[:, :, 0].copy(), g[:, :, 0], p)
        green_image = naive_blending(h[:, :, 1].copy(), g[:, :, 1], p)
        blue_image = naive_blending(h[:, :, 2].copy(), g[:, :, 2], p)
    elif blend_type == 1:
        red_image = laplace_blending(h[:, :, 0].copy(), g[:, :, 0], p)
        green_image = laplace_blending(h[:, :, 1].copy(), g[:, :, 1], p)
        blue_image = laplace_blending(h[:, :, 2].copy(), g[:, :, 2], p)
    elif blend_type == 2:
        red_image = mix_blend(h[:, :, 0].copy(), g[:, :, 0], p)
        green_image = mix_blend(h[:, :, 1].copy(), g[:, :, 1], p)
        blue_image = mix_blend(h[:, :, 2].copy(), g[:, :, 2], p)
    else:
        raise ValueError("The given blend type does not exist. Must be between 0 - 2.")

    image = np.stack((red_image, green_image, blue_image), axis=2)
    return image


def plane_bird_example(blend_type):
    # Read in the images and set pixel offset `p`
    h = io.imread("input/bird.jpg")
    g = io.imread("input/plane.jpg")
    p = (400, 50)

    if blend_type == 0:
        red_image = naive_blending(h[:, :, 0].copy(), g[:, :, 0], p)
        green_image = naive_blending(h[:, :, 1].copy(), g[:, :, 1], p)
        blue_image = naive_blending(h[:, :, 2].copy(), g[:, :, 2], p)
    elif blend_type == 1:
        red_image = laplace_blending(h[:, :, 0].copy(), g[:, :, 0], p)
        green_image = laplace_blending(h[:, :, 1].copy(), g[:, :, 1], p)
        blue_image = laplace_blending(h[:, :, 2].copy(), g[:, :, 2], p)
    elif blend_type == 2:
        red_image = mix_blend(h[:, :, 0].copy(), g[:, :, 0], p)
        green_image = mix_blend(h[:, :, 1].copy(), g[:, :, 1], p)
        blue_image = mix_blend(h[:, :, 2].copy(), g[:, :, 2], p)
    else:
        raise ValueError("The given blend type does not exist. Must be between 0 - 2.")

    image = np.stack((red_image, green_image, blue_image), axis=2)
    return image


def main():
    print(
        "WARNING: This will take some time. The program will compute a lot of redundant matrices."
    )
    # Define values for displaying the blended images
    col = 3
    row = 2
    images = []
    titles = [
        [
            "Naive Blend - Bear in Water",
            "Laplace Blend - Bear in Water",
            "Mix Blend - Bear in Water",
        ],
        [
            "Naive Blend - Plane and Bird",
            "Laplace Blend - Plane and Bird",
            "Mix Blend - Plane and Bird",
        ],
    ]
    for i in range(3):
        images.append(bear_water_example(i))
        print(f"Generated: {titles[0][i]}")
        images.append(plane_bird_example(i))
        print(f"Generated: {titles[1][i]}")

    # Create a figure and subplots based on the number of images
    fig, axes = plt.subplots(row, col, figsize=(col * 5, row * 5))

    # Blend all images and append to `images`
    for i in range(len(images)):
        row = i % 2
        col = i // 2
        axes[row, col].imshow(images[i])
        axes[row, col].set_title(titles[row][col])
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
