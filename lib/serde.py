import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_results_from_directory(dir_path: Path):
    """Given a directory, load all CSVs and return lists of x and y arrays."""

    dfs = [
        pd.read_csv(path, index_col="num_evaluations")
        for path in glob.glob(f"{dir_path}/*.csv")
    ]
    common_idx = sorted(set().union(*[df.index for df in dfs]))
    dfs_interp = [df.reindex(common_idx).interpolate(method="index") for df in dfs]

    stacked = np.stack([df.values for df in dfs_interp])
    mean = stacked.mean(axis=0)
    return pd.DataFrame(
        mean,
        index=common_idx,  # pyright: ignore[reportArgumentType]
        columns=dfs[0].columns,
    )
