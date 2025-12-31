import multiprocessing as mp
import os
from dataclasses import dataclass

import pandas as pd
from loguru import logger

import lib.metrics as m
from experiments.find_switch_interval.cmaes.experiment_config import (
    CMAESExperimentConfig,
    ObjectiveChoice,
    OptimumPosition,
)
from lib.metrics_collector import MetricsCollector
from lib.optimizers.cmaes import CMAES
from lib.random import IndividualGenerator
from lib.stopping import CMAESEarlyStopping
from lib.util import EvalCounter, compress_and_save, summarize_data


@dataclass
class CMAESExperiment:
    config: CMAESExperimentConfig

    def run_subprocess(self, run_id: int):
        objective = self.config.get_objective_instance()
        counter = EvalCounter(objective, bounds=self.config.bounds)  # pyright: ignore[reportArgumentType]
        rng = IndividualGenerator(run_id, self.config.bounds, self.config.dimensions)
        x0 = rng.get_individual()

        collector = MetricsCollector(
            [
                m.CMAESIteration(self.config.popsize),
                m.CovarianceMatrix(serialize=True),
                m.BestSoFar(),
                m.Mean(),
            ],
            run_id,
            every_n_calls=self.config.collection_interval,
        )

        cmaes = CMAES(
            counter,
            x0,
            self.config.popsize,
            rng.seed,
            CMAESEarlyStopping(
                self.config.max_evals,
                tolfun=1e-9,
            ),
            collector,
            self.config.bounds,
        )
        logger.info(f"{run_id}: constructed CMA-ES, starting optimization")
        cmaes.optimize()
        logger.info(f"{run_id}: done with optimization")
        return collector.as_dataframe()

    def archive_data(self, data: list[pd.DataFrame]):
        df = pd.concat(data)
        outpath = self.config.output_directory / "raw.parquet"
        compress_and_save(df, outpath)

        if self.config.debug:
            df = pd.read_parquet(outpath)
            summarize_data(df)

    def run(self):
        with mp.Pool(mp.cpu_count()) as pool:
            run_indices = self.config.get_run_indices()
            data = pool.map(self.run_subprocess, run_indices)
            self.archive_data(data)


if __name__ == "__main__":
    logger.remove()
    debug = bool(os.getenv("DEBUG", ""))
    print(f"Debug mode: {debug}")

    if debug:
        config = CMAESExperimentConfig(
            10, 5, ObjectiveChoice.ELLIPTIC, OptimumPosition.MIDDLE, True
        )
    else:
        config = CMAESExperimentConfig.create_from_env()

    print(f"Configuration: {config}")

    experiment = CMAESExperiment(config)
    experiment.run()
