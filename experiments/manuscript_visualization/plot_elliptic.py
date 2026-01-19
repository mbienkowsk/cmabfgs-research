from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOT_PATH = Path(__file__).parent / "plots" / "elliptic_contour.png"


def elliptic_vectorized(x):
    n = len(x)
    return sum(10 ** (6 * i / (n - 1)) * x[i] ** 2 for i in range(n))


if __name__ == "__main__":
    RANGE = 100
    x1 = np.linspace(-RANGE, RANGE, 400)
    x2 = np.linspace(-RANGE, RANGE, 400)
    X1, X2 = np.meshgrid(x1, x2)

    Z = elliptic_vectorized([X1, X2])

    plt.figure()
    plt.contourf(X1, X2, Z, levels=30)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
