from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd


def tex(text: str):
    return f"${text}$"


def configure_mpl_for_manuscript():
    import matplotlib as mpl

    mpl.rcParams["font.size"] = 24
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["lines.linewidth"] = 3


def set_log_x_labels(ax: plt.Axes):
    ax.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )


def plot_with_legend_function(
    df: pd.DataFrame, ax: plt.Axes, legend_fn: Callable[[str], str]
):
    df.plot(ax=ax)
    plt.legend([legend_fn(col) for col in df.columns])
