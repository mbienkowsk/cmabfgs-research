import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ONLY_SHOW = True


def get_plot_directory(dim: int):
    return Path(__file__).parent / "results" / "plots" / f"d_{dim}"


def get_result_directory(dim: int):
    return Path(__file__).parent / "results" / f"d{dim}" / "agg"


def short_label(col: str) -> str:
    m = re.match(r"(best_\d+)", col)
    if not m:
        return col
    label = m.group(1)
    if "normalized" in col:
        label += "_norm"
    return label


def trim_constant_tail(s: pd.Series) -> pd.Series:
    v = s.values
    # find last index where value changes
    idx = (v != v[-1]).nonzero()[0]
    if len(idx) == 0:
        return s.iloc[:1]
    return s.iloc[: idx[-1] + 2]


def visualize_results(df: pd.DataFrame, save_dir: Path, only_show: bool):
    fig, ax = plt.subplots(figsize=(16, 9))
    print(df.columns)
    df = pd.concat(
        {c: trim_constant_tail(df[c]) for c in df.columns},
        axis=1,
    )
    num_cols = df.shape[1]
    normalized_cols = [col for col in df.columns if "normalized" in col]
    raw_cols = [col for col in df.columns if "normalized" not in col]

    df.plot(
        ax=ax,
        y=raw_cols,
        title="BFGS inicjowany macierzą kowariancji CMAESa po n iteracjach - porównanie krzywych zbieżności",
        logy=True,
    )
    plt.legend([short_label(col) for col in raw_cols])
    plt.grid()
    if only_show:
        plt.show()
    else:
        plt.savefig(save_dir / "non_normalized.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(16, 9))
    df.plot(
        ax=ax,
        y=normalized_cols,
        title="BFGS inicjowany znormalizowaną macierzą kowariancji CMAESa po n iteracjach - porównanie krzywych zbieżności",
        logy=True,
    )
    plt.legend([short_label(col) for col in normalized_cols])
    plt.grid()
    if only_show:
        plt.show()
    else:
        plt.savefig(save_dir / "normalized.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    DIMS = [10, 20, 50, 100]

    for dim in DIMS:
        save_dir = get_plot_directory(dim)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "random_x0").mkdir(parents=True, exist_ok=True)
        (save_dir / "inherited_x0").mkdir(parents=True, exist_ok=True)

        save_dir = get_plot_directory(dim)
        visualize_results(
            pd.read_parquet(get_result_directory(dim) / "bfgs.parquet"),
            save_dir / "random_x0",
            ONLY_SHOW,
        )
        visualize_results(
            pd.read_parquet(get_result_directory(dim) / "bfgs_inherited_x0.parquet"),
            save_dir / "inherited_x0",
            ONLY_SHOW,
        )
