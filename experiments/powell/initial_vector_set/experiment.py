from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from scipy.stats import ortho_group

from config.paths import CONFIG_DIR_STR
from config.schema import MasterConfig
from lib.funs import elliptic
from lib.metrics import BestSoFar
from lib.metrics_collector import MetricsCollector
from lib.optimizers.powell import Powell
from lib.random import IndividualGenerator
from lib.stopping import PowellEarlyStopping
from lib.util import EvalCounter, compress_and_save, run_indices_pgbar


@dataclass
class PowellInitialVectorSetConfig:
    dimensions: int
    bounds: float
    num_matrices: int

    @property
    def bounds_tuple(self) -> tuple[float, float]:
        return (-self.bounds, self.bounds)

    @property
    def result_dir(self) -> Path:
        return Path(__file__).parent / "results" / f"d{self.dimensions}"


def rotate_input(fun, R: np.ndarray):
    return lambda x: fun(R @ x)


@dataclass
class PowellInitialVectorSetExperiment:
    cfg: PowellInitialVectorSetConfig

    def generate_matrices(self) -> list[np.ndarray]:
        return [
            ortho_group.rvs(self.cfg.dimensions, random_state=i)
            for i in run_indices_pgbar(self.cfg.num_matrices, "Generating matrices...")
        ]

    def run_worker(self, matrix_id: int, R: np.ndarray) -> pd.DataFrame:
        x0 = IndividualGenerator(
            matrix_id, self.cfg.bounds_tuple, self.cfg.dimensions
        ).get_individual()
        rotated_fn = rotate_input(elliptic, R)
        collector = MetricsCollector([BestSoFar()], run_id=matrix_id)

        for identifier, direc0 in [("default", None), ("rotated_direc", R)]:
            Powell(
                x0=x0.copy(),
                fun=EvalCounter(rotated_fn),
                callback=collector,
                stopper=PowellEarlyStopping(),
                bounds=self.cfg.bounds_tuple,
                direc0=direc0,
                identifier=identifier,
            ).optimize()

        Powell(
            x0=x0.copy(),
            fun=EvalCounter(elliptic),
            callback=collector,
            stopper=PowellEarlyStopping(),
            bounds=self.cfg.bounds_tuple,
            identifier="no_rotation",
        ).optimize()

        return collector.as_dataframe()

    def run(self):
        matrices = self.generate_matrices()
        dfs = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.run_worker)(i, R) for i, R in enumerate(matrices, start=1)
        )
        raw = pd.concat(dfs)
        self.cfg.result_dir.mkdir(parents=True, exist_ok=True)
        compress_and_save(raw, self.cfg.result_dir / "raw.parquet")
        logger.info(f"Saved results to {self.cfg.result_dir}")


@hydra.main(version_base=None, config_name="config", config_path=CONFIG_DIR_STR)
def main(cfg: MasterConfig):
    exp_cfg = cfg.experiments.powell_initial_vector_set
    config = PowellInitialVectorSetConfig(
        dimensions=exp_cfg.dimensions,
        bounds=exp_cfg.bounds,
        num_matrices=exp_cfg.num_matrices,
    )
    PowellInitialVectorSetExperiment(config).run()


if __name__ == "__main__":
    main()
