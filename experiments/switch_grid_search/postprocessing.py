import glob
import re
from pathlib import Path
from typing import cast

import pandas as pd
from loguru import logger
from pandas import DataFrame

from experiments.switch_grid_search.switch_grid_search import visualize_results


def load_from_disk(path: Path) -> tuple[DataFrame, DataFrame]:
    df = pd.read_csv(path, index_col=0)

    df.index.name = "num_evaluations"

    only_bfgs = df["bfgs_best"].to_frame().dropna()
    logger.info(only_bfgs.head())
    cmabfgs = df[df.columns.difference(["bfgs_best"])].dropna(how="all")

    return cast(DataFrame, only_bfgs), cast(DataFrame, cmabfgs)


def redraw_plots():
    g = Path(__file__).parent / "results" / "CEC*_100_combined.csv"
    logger.info(f"looking in {g}")
    files = glob.glob(str(g))
    logger.info(f"Found files: {files}")
    for file in files:
        bfgs, cmabfgs = load_from_disk(Path(file))
        logger.info(cmabfgs.columns)
        steps = sorted(
            [
                int(results.group(1))
                for c in cmabfgs.columns
                if (results := re.search(r"best_(\d+)$", c)) is not None
            ]
        )
        assert steps
        logger.info(steps)
        results = re.search(r"CEC(\d+)_100_combined.csv", Path(file).name)
        try:
            assert results is not None
        except Exception as e:
            logger.error(f"No function number found in {Path(file).name}")
            raise e
        fun_number = int(results.group(1))
        logger.info(bfgs.head())
        logger.info(cmabfgs.head())
        visualize_results(
            bfgs,  # pyright: ignore[reportArgumentType]
            cmabfgs,  # pyright: ignore[reportArgumentType]
            dimensions=100,
            switch_after_iterations=steps,
            function_name=f"CEC{fun_number}",
        )
