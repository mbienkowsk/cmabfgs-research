import re
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

SHOW = False

MANUSCRIPT = True  # hide titles and make text bigger for the manuscript

if MANUSCRIPT:
    mpl.rcParams["font.size"] = 24
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["lines.linewidth"] = 3

colors = plt.cm.tab20.colors  # pyright: ignore[reportAttributeAccessIssue]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)


def get_plot_directory(dim: int):
    return Path(__file__).parent / "results" / "plots" / "new" / f"d_{dim}"


def get_result_directory(dim: int):
    return Path(__file__).parent / "results" / f"d{dim}" / "agg"


def short_label(col: str) -> str:
    m = re.search(r"\d+", col)
    if not m:
        if "identity" in col:
            return "I"
        elif "inv_hess" in col:
            return "H_inv"
        return col
    label = m.group(0)
    return label


def trim_constant_tail(
    s: pd.Series, epsilon: float = 1e-11, leave_first_n: int = 3
) -> pd.Series:
    v = s.values
    idx = (abs(v - v[-1]) >= epsilon).nonzero()[0]
    if len(idx) == 0:
        return s.iloc[:1]
    return s.iloc[: idx[-1] + 2 + leave_first_n]


@contextmanager
def wrap_convergence_plot(
    save_to: Path,
    show: bool = True,
    xlabel: str = "Liczba ewaluacji funkcji celu",
    ylabel: str = "f(xbest)",
    title: str | None = None,
):
    fig, ax = plt.subplots(figsize=(16, 9))

    ax.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    yield ax

    if title is not None and not MANUSCRIPT:
        ax.set_title(title)

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    plt.savefig(save_to, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def visualize_results(df: pd.DataFrame, save_dir: Path, show: bool):
    df = pd.concat(
        {c: trim_constant_tail(df[c]) for c in df.columns},
        axis=1,
    )

    raw_cols = [c for c in df.columns if "normalized" not in c]
    normalized_cols = [c for c in df.columns if "normalized" in c]

    with wrap_convergence_plot(save_dir / "non_normalized.png", show) as ax:
        df.plot(ax=ax, y=raw_cols)
        ax.legend([short_label(c) for c in raw_cols])

    with wrap_convergence_plot(save_dir / "normalized.png", show) as ax:
        df.plot(ax=ax, y=normalized_cols)
        ax.legend([short_label(c) for c in normalized_cols])


if __name__ == "__main__":
    DIMS = [10, 20, 50, 100]
    # DIMS = [10]

    for dim in tqdm(DIMS, "Processing dimension sets...", total=len(DIMS)):
        save_dir = get_plot_directory(dim)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "random_x0").mkdir(parents=True, exist_ok=True)
        (save_dir / "inherited_x0").mkdir(parents=True, exist_ok=True)

        save_dir = get_plot_directory(dim)
        visualize_results(
            pd.read_parquet(get_result_directory(dim) / "bfgs.parquet"),
            save_dir / "random_x0",
            SHOW,
        )
        # visualize_results(
        #     pd.read_parquet(get_result_directory(dim) / "bfgs_inherited_x0.parquet"),
        #     save_dir / "inherited_x0",
        #     ONLY_SHOW,
        # )
