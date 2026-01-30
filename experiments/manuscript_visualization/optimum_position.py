from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.find_switch_interval.common import OptimumPosition
from lib.plotting_util import configure_mpl_for_manuscript

F_GLOBAL = 0.0
PLOT_PADDING = 25.0

USE_SINGLE_PLOT = True

sns.set_theme(style="whitegrid")


def to_plot_title(pos: OptimumPosition):
    match pos:
        case OptimumPosition.MIDDLE:
            return "Przypadek 1 – optimum w środku\n obszaru dopuszczalnego"
        case OptimumPosition.CORNER:
            return (
                "Przypadek 2 – optimum w rogu\n obszaru dopuszczalnego (margines 10%)"
            )
        case OptimumPosition.CORNER_NEAR:
            return (
                "Przypadek 3 – optimum w rogu\n obszaru dopuszczalnego (margines 0.5%)"
            )
        case OptimumPosition.OUTSIDE_CORNER:
            return "Przypadek 4 – optimum poza rogiem\n obszaru dopuszczalnego"


def to_label(pos: OptimumPosition):
    """Shorter labels for single plot legend"""
    match pos:
        case OptimumPosition.MIDDLE:
            return "1. Środek obszaru"
        case OptimumPosition.CORNER:
            return "2. Róg (margines 10%)"
        case OptimumPosition.CORNER_NEAR:
            return "3. Róg (margines 0.5%)"
        case OptimumPosition.OUTSIDE_CORNER:
            return "4. Poza rogiem"


def draw_case(data: pd.DataFrame, **kwargs):
    ax = plt.gca()
    low = data["low"].iloc[0]
    high = data["high"].iloc[0]
    ax.plot(
        [low, high, high, low, low],
        [low, low, high, high, low],
        color="black",
        linewidth=2,
    )
    ax.scatter(
        F_GLOBAL,
        F_GLOBAL,
        color="red",
        s=40,
        label="Położenie optimum globalnego",
        zorder=3,
    )
    ax.set_xlim(low - PLOT_PADDING, high + PLOT_PADDING)
    ax.set_ylim(low - PLOT_PADDING, high + PLOT_PADDING)
    ax.set_aspect("equal", adjustable="box")


if __name__ == "__main__":
    configure_mpl_for_manuscript()

    rows = []
    for p in OptimumPosition:
        low, high = p.get_bounds()
        rows.append(
            {
                "case": to_plot_title(p),
                "label": to_label(p),
                "low": low,
                "high": high,
                "position": p,
            }
        )
    df = pd.DataFrame(rows)

    if USE_SINGLE_PLOT:
        # Single plot version
        fig, ax = plt.subplots(figsize=(8, 8))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for idx, row in df.iterrows():
            low = row["low"]
            high = row["high"]

            ax.plot(
                [low, high, high, low, low],
                [low, low, high, high, low],
                color=colors[idx],
                linewidth=2,
                label=row["label"],
            )

        ax.scatter(
            F_GLOBAL,
            F_GLOBAL,
            color="red",
            s=100,
            marker="*",
            label="Optimum globalne",
            zorder=3,
            edgecolors="darkred",
            linewidths=1.5,
        )

        ax.set_xlim(-245, 225)
        ax.set_ylim(-245, 125)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("Warianty położenia ograniczeń względem optimum globalnego")

        ax.legend(loc="best", framealpha=0.95)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

    else:
        # Original 4-panel version
        g = sns.FacetGrid(
            df,
            col="case",
            col_wrap=2,
            height=4.8,
            aspect=1,
            sharex=False,
            sharey=False,
        )
        g.map_dataframe(draw_case)
        handles, labels = g.axes.flat[0].get_legend_handles_labels()
        g.fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
        )
        g.fig.subplots_adjust(top=0.88)
        g.fig.set_size_inches(12, 12)
        g.set_titles("{col_name}")

    plt.savefig(
        Path(__file__).parent / "plots" / "optimum_positions.png",
        dpi=300,
        bbox_inches="tight",
    )
