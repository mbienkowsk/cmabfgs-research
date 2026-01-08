from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.find_switch_interval.common import OptimumPosition

F_GLOBAL = 0.0
PLOT_PADDING = 25.0

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
    rows = []
    for p in OptimumPosition:
        low, high = p.get_bounds()
        rows.append(
            {
                "case": to_plot_title(p),
                "low": low,
                "high": high,
            }
        )

    df = pd.DataFrame(rows)

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
    g.fig.suptitle(
        "Warianty ograniczeń kostkowych zastosowanych w badaniu",
        fontsize=14,
        y=0.995,
    )
    g.fig.subplots_adjust(top=0.90)

    plt.savefig(
        Path(__file__).parent / "plots" / "optimum_positions.png",
        dpi=300,
        bbox_inches="tight",
    )
