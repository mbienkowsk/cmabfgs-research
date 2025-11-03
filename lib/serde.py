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
    """aggregate dataframes with the same columns where
    num_evaluations is the index"""
    common_idx = sorted(set().union(*[df.index for df in dfs]))

    stacked = np.stack([df.values for df in dfs])
    mean = stacked.mean(axis=0)
    return pd.DataFrame(
        mean,
        index=common_idx,  # pyright: ignore[reportArgumentType]
        columns=next(iter(dfs)).columns,
    )
