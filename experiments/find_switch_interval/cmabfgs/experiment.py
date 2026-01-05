import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from typing import override

import numpy as np
import pandas as pd
from loguru import logger

import lib.metrics as m
from experiments.find_switch_interval.cmabfgs.experiment_config import (
    CMABFGSExperimentConfig,
)
from experiments.find_switch_interval.common import (
    ExperimentBase,
    HessianNormalization,
    ObjectiveChoice,
    OptimumPosition,
)
from lib.metrics_collector import MetricsCollector
from lib.optimizers.bfgs import BFGS
from lib.random import IndividualGenerator
from lib.stopping import BFGSEarlyStopping
from lib.util import EvalCounter, compress_and_save, summarize_data


@dataclass
class CMABFGSExperiment(ExperimentBase[CMABFGSExperimentConfig]):
    config: CMABFGSExperimentConfig

    def run_vanilla_bfgs(self, run_id: int, collector: MetricsCollector):
        rng = IndividualGenerator(run_id, self.config.bounds, self.config.dimensions)
        counter = EvalCounter(
            self.config.get_objective_instance(),  # pyright: ignore[reportArgumentType]
            bounds=self.config.bounds,
            kill_outside_bounds=True,
        )
        x0 = rng.get_individual()
        bfgs = BFGS(
            x0,
            counter,
            collector,
            BFGSEarlyStopping(max_evals=self.config.max_evals),
            self.config.bounds,
            identifier="vanilla_bfgs",
        )
        bfgs.optimize()
        return

    def reconstruct_covariance_matrix(self, mat: np.ndarray):
        reshaped = np.reshape(mat, (self.config.dimensions, self.config.dimensions))
        # normalize to norm=dim
        normalized = reshaped / np.linalg.norm(reshaped)
        if self.config.hess_normalization == HessianNormalization.UNIT_DIM:
            normalized *= self.config.dimensions

        return normalized * 0.5 + normalized.T * 0.5

    def run_subprocess(self, run_id: int, df: pd.DataFrame):
        collector = MetricsCollector([m.BestSoFar()], run_id)
        self.run_vanilla_bfgs(run_id, collector)

        for iters, row in df[df["cov_mat"].notna()].set_index("iteration").iterrows():
            counter = EvalCounter(
                self.config.get_objective_instance(),  # pyright: ignore[reportArgumentType]
                bounds=self.config.bounds,
                kill_outside_bounds=True,
            )
            bfgs = BFGS(
                row["mean"],
                counter,
                collector,
                BFGSEarlyStopping(max_evals=self.config.max_evals),
                self.config.bounds,
                identifier=str(iters),
                hess_inv0=self.reconstruct_covariance_matrix(row["cov_mat"]),
            )
            bfgs.optimize()

        # join column-wise for single df with cmaes + bfgs
        rv = pd.concat(
            [df["best"].rename("best_cmaes"), collector.as_dataframe()], axis=1
        )
        rv["run_id"] = run_id  # fill empty cmaes values
        return rv

    def archive_data(self, data: list[pd.DataFrame]):
        raw = pd.concat(data)
        outpath = self.config.output_directory / "raw.parquet"
        compress_and_save(raw, outpath)

        if self.config.debug:
            df = pd.read_parquet(outpath)
            summarize_data(df)

    @override
    def run(self):
        input_df = pd.read_parquet(self.config.input_file)
        per_run = [df for _, df in input_df.groupby("run_id")]
        assert len(per_run) == self.config.num_runs, (
            f"Number of runs in input does not match configuration (expected {len(per_run)}, found {self.config.num_runs})"
        )

        with mp.Pool(mp.cpu_count()) as pool:
            run_indices = self.config.get_run_indices()
            data = pool.starmap(self.run_subprocess, zip(run_indices, per_run))
            self.archive_data(data)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    debug = bool(os.getenv("DEBUG", ""))
    print(f"Debug mode: {debug}")

    if debug:
        config = CMABFGSExperimentConfig(
            10,
            25,
            ObjectiveChoice.ELLIPTIC,
            OptimumPosition.MIDDLE,
            True,
            HessianNormalization.UNIT_DIM,
        )
    else:
        config = CMABFGSExperimentConfig.create_from_env()

    print(f"Configuration: {config}")

    experiment = CMABFGSExperiment(config)
    experiment.run()
