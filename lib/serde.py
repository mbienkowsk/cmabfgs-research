import glob
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def load_results_from_directory(dir_path: Path):
    """Given a directory, load all CSVs and return lists of x and y arrays."""

    dfs = [
        pd.read_csv(path, index_col="num_evaluations")
        for path in glob.glob(f"{dir_path}/*.csv")
    ]
    return aggregate_dataframes(dfs)


def aggregate_dataframes(
    dfs: Iterable[pd.DataFrame],
    drop_col: str | None = "run_id",
    add_quartiles: bool = False,
) -> pd.DataFrame:
    if drop_col is not None:
        dfs = [df.drop(columns=[drop_col]) for df in dfs]

    common_index = np.unique(np.concatenate([df.index.values for df in dfs]))  # pyright: ignore[reportCallIssue, reportArgumentType]

    aligned = [
        df.reindex(common_index).interpolate(method="index", limit_direction="both")
        for df in dfs
    ]

    stacked = pd.concat(aligned).groupby(level=0)

    mean_df = stacked.mean()

    if not add_quartiles:
        return mean_df  # pyright: ignore[reportReturnType]

    q25 = stacked.quantile(0.25).add_suffix("_q25")
    q75 = stacked.quantile(0.75).add_suffix("_q75")

    return pd.concat([mean_df, q25, q75], axis=1)  # pyright: ignore[reportCallIssue, reportArgumentType]
