import sys
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from scipy.stats import ortho_group

from config.paths import CMAES_C_METRICS_DIR, CONFIG_DIR_STR
from config.schema import MasterConfig
from lib.funs import get_function_by_name, rotate_input
from lib.metrics import BestSoFar
from lib.metrics_collector import MetricsCollector
from lib.optimizers.powell import Powell
from lib.random import IndividualGenerator
from lib.stopping import PowellEarlyStopping
from lib.util import EvalCounter, compress_and_save, evaluation_budget


@dataclass
class PowellCMatrixConfig:
    dimensions: int
    bounds: float
    objective: str
    rot_matrix_idx: int | None
    max_evals: int = field(init=False)

    def __post_init__(self):
        self.max_evals = evaluation_budget(self.dimensions)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    @property
    def bounds_tuple(self) -> tuple[float, float]:
        return (-self.bounds, self.bounds)

    @property
    def _rot_dir(self) -> str:
        return (
            f"rot_{self.rot_matrix_idx}"
            if self.rot_matrix_idx is not None
            else "no_rotation"
        )

    @property
    def cmaes_data_path(self) -> Path:
        return (
            CMAES_C_METRICS_DIR
            / self.objective
            / f"d{self.dimensions}"
            / self._rot_dir
            / "raw.parquet"
        )

    @property
    def result_dir(self) -> Path:
        return (
            Path(__file__).parent
            / "results"
            / self.objective
            / f"d{self.dimensions}"
            / self._rot_dir
        )

    @classmethod
    def from_omegaconf(cls, cfg) -> "PowellCMatrixConfig":
        return cls(
            dimensions=cfg.dimensions,
            bounds=cfg.bounds,
            objective=cfg.objective,
            rot_matrix_idx=cfg.get("rot_matrix_idx", None),
        )


def reconstruct_cov_mat(flat: np.ndarray, dim: int) -> np.ndarray:
    reshaped = np.reshape(flat, (dim, dim))
    return reshaped * 0.5 + reshaped.T * 0.5


@dataclass
class PowellCMatrixExperiment:
    config: PowellCMatrixConfig

    def load_cmaes_data(self) -> pd.DataFrame:
        df = pd.read_parquet(self.config.cmaes_data_path)
        return df[df["cov_mat"].notna()]

    def make_objective(self):
        fn = get_function_by_name(self.config.objective, self.config.dimensions)
        if self.config.rot_matrix_idx is None:
            return fn
        R = ortho_group.rvs(
            self.config.dimensions, random_state=self.config.rot_matrix_idx
        )
        return rotate_input(fn, R)

    def run_powell(
        self, x0: np.ndarray, direc0: np.ndarray, fn, run_id: int
    ) -> pd.DataFrame:
        counter = EvalCounter(fn, bounds=self.config.bounds_tuple)
        collector = MetricsCollector([BestSoFar()], run_id)
        Powell(
            x0=x0.copy(),
            fun=counter,
            callback=collector,
            stopper=PowellEarlyStopping(self.config.max_evals),
            bounds=self.config.bounds_tuple,
            direc0=direc0.copy(),
        ).optimize()
        return collector.as_dataframe().reset_index()

    def run_worker(
        self,
        run_id: int,
        cma_num_evaluations: int,
        cov_mat_flat: list,
        mean: list,
    ) -> pd.DataFrame:
        dim = self.config.dimensions
        cov_mat = reconstruct_cov_mat(np.array(cov_mat_flat), dim)
        direc0 = np.linalg.eigh(cov_mat)[1].T
        cma_mean = np.array(mean)
        cma_start = IndividualGenerator(
            run_id, self.config.bounds_tuple, dim
        ).get_individual()
        fn = self.make_objective()

        dfs = []
        for condition, x0 in [("cma_mean", cma_mean), ("cma_start", cma_start)]:
            df = self.run_powell(x0, direc0, fn, run_id)
            df["condition"] = condition
            df["cma_num_evaluations"] = cma_num_evaluations
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def run(self):
        cmaes_df = self.load_cmaes_data()
        logger.info(
            f"Loaded {len(cmaes_df)} CMA-ES rows with covariance matrices across "
            f"{cmaes_df['run_id'].nunique()} runs"
        )

        tasks = [
            (int(row["run_id"]), int(num_evals), row["cov_mat"], row["mean"])
            for num_evals, row in cmaes_df.iterrows()
        ]

        dfs = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.run_worker)(run_id, cma_num_evals, cov_mat, mean)
            for run_id, cma_num_evals, cov_mat, mean in tasks
        )
        raw = pd.concat(dfs, ignore_index=True)  # pyright: ignore[reportCallIssue, reportArgumentType]
        raw = raw.set_index("num_evaluations")
        compress_and_save(raw, self.config.result_dir / "raw.parquet")
        logger.info(
            f"Saved {len(raw)} rows to {self.config.result_dir / 'raw.parquet'}"
        )


@hydra.main(version_base=None, config_name="config", config_path=CONFIG_DIR_STR)
def main(cfg: MasterConfig):
    exp_cfg = cfg.experiments.powell_c_matrix  # ty: ignore[unresolved-attribute]
    config = PowellCMatrixConfig.from_omegaconf(exp_cfg)
    logger.info(f"Config: {config}")
    PowellCMatrixExperiment(config).run()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
