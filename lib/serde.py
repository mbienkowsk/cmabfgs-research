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


def aggregate_dataframes(dfs: Iterable[pd.DataFrame]):
    dfs = [df.drop(columns=["run_id"]) for df in dfs]
    common_index = np.unique(
        np.concatenate([df.index.values for df in dfs])  # pyright: ignore[reportCallIssue, reportArgumentType]
    )
    aligned = [df.reindex(common_index).interpolate(method="index") for df in dfs]
    return pd.concat(aligned).groupby(level=0).mean()
