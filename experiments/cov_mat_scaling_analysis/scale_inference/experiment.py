from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from config.schema import MasterConfig
from data import get_cmaes_c_metrics
from lib.funs import get_function_by_name
from lib.metrics import BestSoFar
from lib.metrics_collector import MetricsCollector
from lib.optimizers import BFGS
from lib.optimizers.bfgs import BFGSState
from lib.stopping import BFGSEarlyStopping
from lib.util import EvalCounter, make_symmetrical, summarize_data


@dataclass
class CScaleInferenceConfig:
    dimensions: int
    objective: str
    bounds: int


@dataclass
class CScaleInferenceExperiment:
    cfg: CScaleInferenceConfig

    @property
    def bfgs_at_iterations(self):
        match self.cfg.dimensions:
            case 10:
                return [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
            case 100:
                return [200, 400, 600, 800, 1000, 1200, 1400, 1600]
            case _:
                raise ValueError(
                    f"bfgs_at_iterations list not provided for dimensionality {self.cfg.dimensions}"
                )

    @property
    def bounds(self):
        return -self.cfg.bounds, self.cfg.bounds

    def get_counter(self):
        return EvalCounter(
            get_function_by_name(self.cfg.objective),  # ty: ignore[invalid-argument-type]
            bounds=(-self.cfg.bounds, self.cfg.bounds),
        )

    @property
    def result_dir(self) -> Path:
        return (
            Path(__file__).parent
            / "results"
            / f"d{self.cfg.dimensions}"
            / f"{self.cfg.objective}_{self.cfg.bounds}"
        )

    def run_bfgs(
        self,
        x0: np.ndarray,
        collector: MetricsCollector,
        hess_inv: np.ndarray,
        identifier: str,
    ) -> BFGSState:
        objective = self.get_counter()
        bfgs = BFGS(
            deepcopy(x0),
            objective,
            collector,
            BFGSEarlyStopping(),
            bounds=self.bounds,
            identifier=identifier,
            hess_inv0=deepcopy(hess_inv),
        )
        bfgs.optimize()
        return bfgs.state

    def single_thread(self, run_id: int, cmaes_data: pd.DataFrame):
        collector = MetricsCollector([BestSoFar()], run_id)
        d = self.cfg.dimensions
        for idx, row in cmaes_data.iterrows():
            iters = row["iteration"]
            unit_eye = np.eye(d) / np.sqrt(d)
            state = self.run_bfgs(
                row["mean"],
                collector,
                unit_eye,
                f"{iters}_identity_unit",
            )

            if state.end_result is None:
                logger.error(
                    f"state.end_result is None for run {run_id}, iters={iters}. Setting scale to 1."
                )
                b_mat_norm = 1
            else:
                b_mat_norm = np.linalg.norm(state.end_result.hess_inv)
            unit_scaled = np.eye(d) / np.sqrt(d) * b_mat_norm
            self.run_bfgs(
                row["mean"],
                collector,
                unit_scaled,
                f"{iters}_identity_rescaled",
            )
            cov_mat = np.array(row["cov_mat"]).reshape((d, d))
            cov_mat_scaled = make_symmetrical(
                cov_mat / np.linalg.norm(cov_mat) * b_mat_norm
            )
            self.run_bfgs(row["mean"], collector, cov_mat_scaled, f"{iters}_C_rescaled")
            self.run_bfgs(
                row["mean"],
                collector,
                make_symmetrical(cov_mat / np.linalg.norm(cov_mat)),
                f"{iters}_C_unit",
            )
        return collector.as_dataframe()

    def run(self):
        cmaes_data = get_cmaes_c_metrics(self.cfg.objective, self.cfg.dimensions)
        rvs = Parallel(n_jobs=-1)(
            delayed(self.single_thread)(
                run_id,
                df.reset_index()[
                    df.reset_index()["iteration"].isin(self.bfgs_at_iterations)
                ],
            )
            for run_id, df in cmaes_data.groupby("run_id")
        )
        df = pd.concat(rvs)

        self.result_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.result_dir / "raw.parquet")
        logger.info("Saved resulting data")
        summarize_data(df)


@hydra.main(version_base=None, config_name="config", config_path="../../../config/")
def main(cfg: MasterConfig):
    CScaleInferenceExperiment(cfg.experiments.c_scale_inference).run()


if __name__ == "__main__":
    main()
