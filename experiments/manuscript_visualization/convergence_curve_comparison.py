from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lib.plotting_util import configure_mpl_for_manuscript

PLOT_DIR = Path(__file__).parent / "plots"

if __name__ == "__main__":
    configure_mpl_for_manuscript()

    N_SAMPLES = 5
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    timepoints = np.logspace(0, 3, 50)
    x = np.arange(0, N_SAMPLES)
    # a clearly better than B
    a_y = np.array([5, 3.5, 2, 1, 0.1])
    b_y = np.array([5, 4, 3, 2.5, 1.8])

    axes[0].plot(x, a_y, label="Optymalizator A", color="blue")
    axes[0].plot(x, b_y, label="Optymalizator B", color="orange")
    axes[0].set_title("A jednoznacznie lepszy od B")
    axes[0].grid()

    # can't be determined
    a_y = np.array([5, 3.5, 2, 1, 0.1])
    b_y = np.array([5, 4.5, 1.6, 0.6, 0.0])

    axes[1].plot(x, a_y, label="Optymalizator A", color="blue")
    axes[1].plot(x, b_y, label="Optymalizator B", color="orange")
    axes[1].set_title("Nie można określić\n lepszego optymalizatora")
    axes[1].grid()
    handles, labels = axes[1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(0.98, -0.05),
        ncol=1,
        frameon=True,
    )

    fig.supxlabel("Liczba ewaluacji f. celu")
    fig.supylabel("f_best")

    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "convergence_curve_comparison.png", dpi=300, bbox_inches="tight"
    )
