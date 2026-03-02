import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from omegaconf import OmegaConf

from lib.bound_handling import BoundEnforcement
from lib.funs import elliptic_hess_inv_for_dim, get_function_by_name
from lib.metrics import BestSoFar
from lib.metrics_collector import MetricsCollector
from lib.optimizers import BFGS
from lib.random import IndividualGenerator
from lib.stopping import BFGSEarlyStopping
from lib.util import EvalCounter, compress_and_save, run_indices_pgbar, summarize_data


@dataclass
class BScaleComparisonExperimentConfig:
    num_runs: int
    dimensions: int
    bounds: tuple[float, float]
    scaling: float | Literal["adaptive"]
    noise: float

    @property
    def result_dir(self):
        return (
            Path(__file__).parent
            / "results"
            / f"d{self.dimensions}"
            / f"bounds_{int(self.bounds[1])}"
            / f"scaling_{self.scaling}"
            / f"noise_{self.noise}"
        )

    @classmethod
    def from_omegaconf(cls, cfg: OmegaConf):
        num_runs, dimensions, bounds, scaling, noise = (
            cfg["num_runs"],  # pyright: ignore[reportIndexIssue]
            cfg["dimensions"],  # pyright: ignore[reportIndexIssue]
            cfg["bounds"],  # pyright: ignore[reportIndexIssue]
            cfg["scaling"],  # pyright: ignore[reportIndexIssue]
            cfg["noise"],  # pyright: ignore[reportIndexIssue]
        )
        return cls(
            num_runs=num_runs,
            dimensions=dimensions,
            bounds=(-bounds, bounds),
            scaling=scaling,
            noise=noise,
        )


@dataclass
class BScaleComparisonExperiment:
    cfg: BScaleComparisonExperimentConfig

    def run_subprocess(self, run_id: int):
        rng = IndividualGenerator(run_id, self.cfg.bounds, self.cfg.dimensions)
        collector = MetricsCollector([BestSoFar()], run_id)
        fn = get_function_by_name("Elliptic")
        counter = EvalCounter(
            fn,  # pyright: ignore[reportArgumentType]
            bounds=self.cfg.bounds,
            bound_enforcement_method=BoundEnforcement.ADDITIVE_PENALTY,
        )
        hess_inv = elliptic_hess_inv_for_dim(self.cfg.dimensions)

        if self.cfg.scaling == "adaptive":
            raise NotImplementedError("Adaptive scaling not implemented yet")

        assert isinstance(self.cfg.scaling, float)
        scaled_h_inv = self.cfg.scaling * hess_inv

        bfgs = BFGS(
            rng.get_individual(),
            counter,
            collector,
            BFGSEarlyStopping(None),
            self.cfg.bounds,
            scaled_h_inv,
        )
        bfgs.optimize()
        return collector.as_dataframe()

    def run(self):
        dfs = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.run_subprocess)(run_id)
            for run_id in run_indices_pgbar(self.cfg.num_runs)
        )
        raw = pd.concat(dfs)  # pyright: ignore[reportCallIssue, reportArgumentType]
        raw["scaling"] = self.cfg.scaling
        compress_and_save(raw, self.cfg.result_dir / "raw.parquet")
        summarize_data(raw)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: OmegaConf):
    config = BScaleComparisonExperimentConfig.from_omegaconf(cfg)
    config.result_dir.mkdir(parents=True, exist_ok=True)
    exp = BScaleComparisonExperiment(config)
    exp.run()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    main()
