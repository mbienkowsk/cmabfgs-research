import sys
from glob import glob
from pathlib import Path

import pandas as pd
from loguru import logger

from config import BASE_DIR


def validate_dataframe(df: pd.DataFrame):
    if df.min().min() >= 0:
        return True, ""

    min_by_run_id = df.groupby("run_id").min()
    negative_cols_by_run_id = {
        idx: min_by_run_id.loc[idx].where(lambda x: x < 0).dropna().index.tolist()
        for idx in min_by_run_id.index
    }
    messages = [
        f"Run {run_id}: {', '.join(cols)}"
        for run_id, cols in negative_cols_by_run_id.items()
        if cols
    ]
    return False, "\n".join(messages)


def validate_results(parquet_dir: Path):
    files = sorted(glob(str(parquet_dir / "*.parquet")))
    logger.info(f"Found {len(files)} files to validate.")
    for file in files:
        valid, message = validate_dataframe(pd.read_parquet(file))
        if not valid:
            logger.error(f"Validation failed for {file}: {message}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Path not provided, exiting.")
        sys.exit(1)

    dir = BASE_DIR / sys.argv[1] / "data"
    if not dir.exists():
        raise FileNotFoundError(f"Directory {dir} does not exist.")

    logger.info(f"Validating {dir}")
    validate_results(dir)
