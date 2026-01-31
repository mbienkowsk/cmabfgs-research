from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def tex(text: str):
    return f"${text}$"


def set_mpl_font_size(size: int):
    mpl.rcParams["font.size"] = size


def configure_mpl_for_manuscript(font_size: int = 24):
    set_mpl_font_size(font_size)
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["lines.linewidth"] = 3
    colors = plt.cm.tab20.colors  # pyright: ignore[reportAttributeAccessIssue]
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)


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
