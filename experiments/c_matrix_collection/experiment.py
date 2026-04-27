import sys
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from scipy.stats import ortho_group

import lib.metrics as m
from config.paths import CMAES_C_METRICS_DIR, CONFIG_DIR_STR
from config.schema import MasterConfig
from lib.funs import get_function_by_name, rotate_input
from lib.metrics_collector import MetricsCollector
from lib.optimizers.cmaes import CMAES
from lib.random import IndividualGenerator
from lib.stopping import CMAESEarlyStopping
from lib.util import (
    EvalCounter,
    compress_and_save,
    evaluation_budget,
    run_indices_pgbar,
    summarize_data,
)


@dataclass
class CMatrixCollectionConfig:
    dimensions: int
    num_runs: int
    objective: str
    bounds: tuple[float, float]
    rot_matrix_idx: int | None
    max_evals: int = field(init=False)
    popsize: int = field(init=False)
    collection_interval: int = field(init=False)

    def __post_init__(self):
        self.max_evals = evaluation_budget(self.dimensions)
        self.popsize = 4 * self.dimensions
        self.collection_interval = self.dimensions // 2
        self.output_directory.mkdir(parents=True, exist_ok=True)

    @property
    def output_directory(self) -> Path:
        rot_dir = (
            f"rot_{self.rot_matrix_idx}"
            if self.rot_matrix_idx is not None
            else "no_rotation"
        )
        return CMAES_C_METRICS_DIR / self.objective / f"d{self.dimensions}" / rot_dir

    @classmethod
    def from_omegaconf(cls, cfg) -> "CMatrixCollectionConfig":
        rot_matrix_idx = cfg.get("rot_matrix_idx", None)
        return cls(
            dimensions=cfg.dimensions,
            num_runs=cfg.num_runs,
            objective=cfg.objective,
            bounds=(-cfg.bounds, cfg.bounds),
            rot_matrix_idx=rot_matrix_idx,
        )


@dataclass
class CMatrixCollectionExperiment:
    config: CMatrixCollectionConfig

    def run_subprocess(self, run_id: int) -> pd.DataFrame:
        fn = get_function_by_name(self.config.objective, self.config.dimensions)
        if self.config.rot_matrix_idx is not None:
            R = ortho_group.rvs(
                self.config.dimensions, random_state=self.config.rot_matrix_idx
            )
            fn = rotate_input(fn, R)

        counter = EvalCounter(fn, bounds=self.config.bounds)  # ty: ignore[invalid-argument-type]
        rng = IndividualGenerator(run_id, self.config.bounds, self.config.dimensions)
        x0 = rng.get_individual()

        convergence_collector = MetricsCollector(
            [m.CMAESIteration(self.config.popsize), m.BestSoFar()],
            run_id,
            every_n_calls=1,
        )
        cov_mat_collector = MetricsCollector(
            [m.CovarianceMatrix(serialize=True), m.Mean()],
            run_id,
            every_n_calls=self.config.collection_interval,
        )

        cmaes = CMAES(
            counter,
            x0,
            self.config.popsize,
            rng.seed,
            CMAESEarlyStopping(self.config.max_evals, tolfun=1e-9),
            [convergence_collector, cov_mat_collector],
            self.config.bounds,
        )
        logger.info(f"{run_id}: starting CMA-ES")
        cmaes.optimize()
        logger.info(f"{run_id}: done")

        return convergence_collector.as_dataframe().merge(
            cov_mat_collector.as_dataframe(),
            on=["num_evaluations", "run_id"],
            how="outer",
        )

    def run(self):
        dfs = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.run_subprocess)(run_id)
            for run_id in run_indices_pgbar(self.config.num_runs)
        )
        raw = pd.concat(dfs)  # pyright: ignore[reportCallIssue, reportArgumentType]
        compress_and_save(raw, self.config.output_directory / "raw.parquet")
        summarize_data(raw)


@hydra.main(version_base=None, config_name="config", config_path=CONFIG_DIR_STR)
def main(cfg: MasterConfig):
    exp_cfg = cfg.experiments.c_matrix_collection  # ty: ignore[unresolved-attribute]
    config = CMatrixCollectionConfig.from_omegaconf(exp_cfg)
    logger.info(f"Config: {config}")
    CMatrixCollectionExperiment(config).run()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
