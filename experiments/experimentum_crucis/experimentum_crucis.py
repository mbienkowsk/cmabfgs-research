import multiprocessing as mp
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from lib.funs import get_function_by_name
from lib.metrics import BestSoFar, CovarianceMatrixEigenvalueList, SigmaMeasurement
from lib.metrics_collector import MetricsCollector
from lib.optimizers.cmaes import CMAES
from lib.stopping import CMAESEarlyStopping
from lib.util import EvalCounter, get_x0_and_seed_for_run_id

LOG_LEVEL = "ERROR"
DEBUG = os.getenv("DEBUG", True)
logger.info(f"Debug set to {DEBUG}")
BOUNDS = 100
OBJECTIVE_NAME = "Elliptic"

if DEBUG:
    DIMENSIONS = 10
    NUM_RUNS = 10
    EXACT_RUN = None
else:
    DIMENSIONS = int(os.environ["DIMENSIONS"])
    NUM_RUNS = int(os.environ["N_RUNS"])

MAXEVALS = 10_000 * DIMENSIONS
POPULATION_SIZE = 4 * DIMENSIONS

RESULT_DIR = (
    Path(__file__).parent / "results" / f"{OBJECTIVE_NAME.lower()}_d{DIMENSIONS}"
)
PLOT_EXPORT_DIR = RESULT_DIR / "plots"

colors = plt.cm.tab20.colors  # pyright: ignore[reportAttributeAccessIssue]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)


def single_run(run_id: int):
    objective = get_function_by_name(OBJECTIVE_NAME, DIMENSIONS)
    counter = EvalCounter(objective, bounds=(-BOUNDS, BOUNDS))  # pyright: ignore[reportArgumentType]
    x, seed = get_x0_and_seed_for_run_id(run_id, DIMENSIONS, BOUNDS)

    collector = MetricsCollector(
        (
            BestSoFar(),
            CovarianceMatrixEigenvalueList(),
            SigmaMeasurement(),
        ),
        "cmaes",
        run_id,
    )

    cmaes = CMAES(
        counter,
        x,
        POPULATION_SIZE,
        seed,
        CMAESEarlyStopping(
            MAXEVALS,
            1e-9,
        ),
        collector,
        (-BOUNDS, BOUNDS),
    )

    logger.info(f"{run_id}: starting optimization")
    cmaes.optimize()
    logger.info(f"{run_id}: done")

    return collector.as_dataframe()


def main():
    with mp.Pool(mp.cpu_count()) as pool:
        run_indices = (
            range(1, NUM_RUNS + 1)
            if not DEBUG or EXACT_RUN is None
            else range(EXACT_RUN, EXACT_RUN + 1)
        )
        rv = pool.map(single_run, run_indices)

    concatenated = pd.concat(rv)
    concatenated.to_parquet(
        RESULT_DIR / f"{OBJECTIVE_NAME.lower()}_d{DIMENSIONS}.parquet",
        index=True,
        compression="brotli",
    )

    # TODO: make sure mean over arrays is ok
    agg = concatenated.drop(columns="run_id").groupby(level=0).mean()
    print(agg)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL)

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_EXPORT_DIR, exist_ok=True)

    main()
