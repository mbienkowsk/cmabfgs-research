# cmaes + occasional bfgs switching:
# 1) from cmaes' x0 with h_inv = C
# 2) from cmaes' xt with h_inv = C
# 3) without preconditioning in these scenarios??
# every n iterations, without subtracting to align


# run_cmaes: ran 25 times, collects cov matrix, bestsofar and m
# concat and save (with x0!)

# run_bfgs - for each entry, run a bfgs from x0 with each C variant


import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from lib.funs import get_function_by_name
from lib.metrics import BestSoFar, CovarianceMatrix
from lib.metrics_collector import MetricsCollector
from lib.optimizers.bfgs import BFGS
from lib.optimizers.cmaes import CMAES
from lib.stopping import BFGSEarlyStopping, CMAESEarlyStopping
from lib.util import EvalCounter, get_x0_and_seed_for_run_id

LOG_LEVEL = "ERROR"
DEBUG = os.getenv("DEBUG", True)
logger.info(f"Debug set to {DEBUG}")
BOUNDS = 100
OBJECTIVE_NAME = "Elliptic"
KILL_OUTSIDE_BOUNDS = False
BFGS_BOUNDS = (-100, 100) if KILL_OUTSIDE_BOUNDS else (-1e9, 1e9)

if DEBUG:
    DIMENSIONS = 10
    NUM_RUNS = 25
    EXACT_RUN = None
    COLLECT_AT_ITERATIONS = [1, 2, 7, 19, 25, 50, 100, 187, 250, 500, 750]

else:
    DIMENSIONS = int(os.environ["DIMENSIONS"])
    NUM_RUNS = int(os.environ["N_RUNS"])
    COLLECT_AT_ITERATIONS = list(map(int, os.environ["SWITCH_AFTER"].split("-")))

MAXEVALS = 10_000 * DIMENSIONS
POPULATION_SIZE = 4 * DIMENSIONS
RESULT_DIR = Path(__file__).parent / "results" / f"d{DIMENSIONS}"


def run_cmaes(run_id: int):
    objective = get_function_by_name(OBJECTIVE_NAME, DIMENSIONS)
    counter = EvalCounter(objective, bounds=(-BOUNDS, BOUNDS))  # pyright: ignore[reportArgumentType]
    x, seed = get_x0_and_seed_for_run_id(run_id, DIMENSIONS, BOUNDS)
    collector = MetricsCollector([CovarianceMatrix(normalize=True)], "cmaes", run_id)

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
    df = collector.as_dataframe()
    # np.arrays are not json-serializable
    df.attrs = {"x0": x.tolist()}

    return df


def deflatten_cov_mat(arr: np.ndarray):
    """To deserialize from parquet, cov matrices have to be deflattened"""
    num_elements = len(arr)
    root = np.sqrt(num_elements).astype(int)
    return arr.reshape((root, root))


def correct_cov_mat(mat: np.ndarray):
    return mat * 1 / 2 + mat.T * 1 / 2


def single_run(run_id: int):
    cmaes_df = run_cmaes(run_id)
    cmaes_df["iteration"] = cmaes_df.index // POPULATION_SIZE
    entries_of_interest = cmaes_df[cmaes_df["iteration"].isin(COLLECT_AT_ITERATIONS)]
    collector = MetricsCollector(
        [BestSoFar()],
        "bfgs",
        run_id,
    )
    for _, row in entries_of_interest.iterrows():
        hess_inv = correct_cov_mat(deflatten_cov_mat(row["cov_mat"]))

        run_bfgs(
            run_id,
            x0=cmaes_df.attrs["x0"],
            collector=collector,
            hess_inv=hess_inv,
            identifier=str(row["iteration"]),
        )

    run_bfgs(
        run_id,
        x0=cmaes_df.attrs["x0"],
        collector=collector,
        hess_inv=np.eye(DIMENSIONS),
        identifier="identity",
    )

    return cmaes_df, collector.as_dataframe()


def run_bfgs(
    run_id: int,
    x0: np.ndarray,
    collector: MetricsCollector,
    hess_inv: np.ndarray,
    identifier: str,
):
    """Fills the given collector with bfgs data, doesn't return anything on its own"""
    objective = get_function_by_name(OBJECTIVE_NAME, DIMENSIONS)
    counter = EvalCounter(objective, bounds=(-BOUNDS, BOUNDS))  # pyright: ignore[reportArgumentType]
    bfgs = BFGS(
        x0,
        counter,
        collector,
        BFGSEarlyStopping(max_evals=MAXEVALS),
        bounds=BFGS_BOUNDS,
        identifier=identifier,
        hess_inv0=hess_inv,
    )
    bfgs.optimize()


def main():
    with mp.Pool(mp.cpu_count()) as pool:
        run_indices = (
            range(1, NUM_RUNS + 1)
            if not DEBUG or EXACT_RUN is None
            else range(EXACT_RUN, EXACT_RUN + 1)
        )
        rv = pool.map(single_run, run_indices)

    cmaes_dfs, bfgs_dfs = [pair[0] for pair in rv], [pair[1] for pair in rv]
    # need to manually clear this since it errors out on cov_mat array comparison
    for df in cmaes_dfs:
        df.attrs = {}

    merged_attrs = {run_id: cmaes_dfs[run_id].attrs for run_id in range(len(cmaes_dfs))}
    concatenated_cmaes = pd.concat(cmaes_dfs)
    concatenated_cmaes.attrs = merged_attrs
    concatenated_cmaes.to_parquet(
        RESULT_DIR / f"cmaes_d{DIMENSIONS}.parquet",
        index=True,
        compression="brotli",
    )

    concatenated_bfgs = pd.concat(bfgs_dfs)
    concatenated_bfgs.to_parquet(
        RESULT_DIR / f"bfgs_d{DIMENSIONS}.parquet",
        index=True,
        compression="brotli",
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL)

    os.makedirs(RESULT_DIR, exist_ok=True)

    main()
