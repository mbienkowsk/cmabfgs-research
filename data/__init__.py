import pandas as pd

from config.paths import CMAES_C_METRICS_DIR


def get_cmaes_c_metrics(objective_name: str, dim: int):
    path = CMAES_C_METRICS_DIR / objective_name / f"d{dim}" / "raw.parquet"
    if not path.exists():
        raise RuntimeError(
            f"Missing CMA-ES C metric data for {objective_name} in {dim} dimensions."
        )
    return pd.read_parquet(path)
