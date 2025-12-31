import matplotlib.pyplot as plt
import numpy as np


def elliptic_vectorized(x):
    n = len(x)
    return sum(10 ** (6 * i / (n - 1)) * x[i] ** 2 for i in range(n))


if __name__ == "__main__":
    x1 = np.linspace(-1, 1, 400)
    x2 = np.linspace(-1, 1, 400)
    X1, X2 = np.meshgrid(x1, x2)

    Z = elliptic_vectorized([X1, X2])

    plt.figure()
    plt.contour(X1, X2, Z, levels=30)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar()
    plt.savefig("elliptic_contour.png")
