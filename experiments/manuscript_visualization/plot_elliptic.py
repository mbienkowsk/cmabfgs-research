from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from lib.plotting_util import configure_mpl_for_manuscript

PLOT_DIR = Path(__file__).parent / "plots"


def elliptic_vectorized(x):
    n = len(x)
    return sum(10 ** (6 * i / (n - 1)) * x[i] ** 2 for i in range(n))


def rastrigin_vectorized(x, A=2.0):
    n = len(x)
    y = np.array(x)

    return A * n + np.sum(y**2 - A * np.cos(2 * np.pi * y), axis=0)


def plot_function_contour(
    func: Callable,
    filename: str,
    x1_range=(-100, 100),
    x2_range=(-100, 100),
    levels=30,
    resolution=400,
    font_size=12,
):
    configure_mpl_for_manuscript(font_size=font_size)

    x1 = np.linspace(x1_range[0], x1_range[1], resolution)
    x2 = np.linspace(x2_range[0], x2_range[1], resolution)
    X1, X2 = np.meshgrid(x1, x2)

    Z = func([X1, X2])

    plt.figure()

    cf = plt.contourf(X1, X2, Z, levels=levels, cmap="viridis")
    plt.contour(X1, X2, Z, levels=20, colors="white", linewidths=0.5, alpha=0.6)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.colorbar(cf)

    plt.tight_layout()

    output_path = PLOT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    plot_function_contour(elliptic_vectorized, "elliptic_contour.png")
    plot_function_contour(
        rastrigin_vectorized,
        "rastrigin_contour.png",
        (-5.12, 5.12),
        (-5.12, 5.12),
        levels=20,
    )
