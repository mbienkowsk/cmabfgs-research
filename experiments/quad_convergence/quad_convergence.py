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
from lib.metrics import BestSoFar, BestXSoFar, CovarianceMatrix
from lib.metrics_collector import MetricsCollector
from lib.optimizers.bfgs import BFGS
from lib.optimizers.cmaes import CMAES
from lib.serde import aggregate_dataframes
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
    NUM_RUNS = 10
    EXACT_RUN = None
    COLLECT_AT_ITERATIONS = [1, 2, 7, 19, 25, 50, 100, 187, 250, 500, 750]

else:
    DIMENSIONS = int(os.environ["DIMENSIONS"])
    NUM_RUNS = int(os.environ["N_RUNS"])
    COLLECT_AT_ITERATIONS = list(map(int, os.environ["SWITCH_AFTER"].split("-")))

MAXEVALS = 10_000 * DIMENSIONS
POPULATION_SIZE = 4 * DIMENSIONS
RESULT_DIR = Path(__file__).parent / "results" / f"d{DIMENSIONS}"
X0_GENERATOR_SEED = 7


def run_cmaes(run_id: int):
    objective = get_function_by_name(OBJECTIVE_NAME, DIMENSIONS)
    counter = EvalCounter(objective, bounds=(-BOUNDS, BOUNDS))  # pyright: ignore[reportArgumentType]
    x, seed = get_x0_and_seed_for_run_id(run_id, DIMENSIONS, BOUNDS)
    collector = MetricsCollector(
        [CovarianceMatrix(normalize=True), BestXSoFar()], "cmaes", run_id
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
    df = collector.as_dataframe()

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
    num_x0s = len(COLLECT_AT_ITERATIONS)
    rng = np.random.default_rng(X0_GENERATOR_SEED)
    x0s = rng.uniform(low=-BOUNDS, high=BOUNDS, size=(num_x0s, DIMENSIONS)).reshape(
        (num_x0s, DIMENSIONS)
    )
    hess_invs = {
        iters: correct_cov_mat(
            deflatten_cov_mat(
                cmaes_df[cmaes_df["iteration"] == iters].iloc[0]["cov_mat"]
            )
        )  # pyright: ignore[reportArgumentType]
        for iters in filter(
            lambda num: num < cmaes_df["iteration"].max(), COLLECT_AT_ITERATIONS
        )
    }
    bfgs_results_raw = [
        run_bfgs_variants_for_x0(run_id, x0s[i], hess_invs).assign(x0_idx=i + 1)
        for i in range(len(x0s))
    ]
    bfgs_index_superset = np.unique(
        np.concatenate([df.index.values for df in bfgs_results_raw])  # pyright: ignore[reportCallIssue, reportArgumentType]
    )
    bfgs_df_agg = (
        pd.concat(
            [
                df.drop(columns="x0_idx")
                .reindex(bfgs_index_superset)
                .interpolate(method="index")
                for df in bfgs_results_raw
            ]
        )
        .groupby(level=0)
        .mean()
        .assign(run_id=run_id)
    )

    bfgs_raw_concat = pd.concat(bfgs_results_raw)
    return cmaes_df, bfgs_df_agg, bfgs_raw_concat


def run_bfgs_variants_for_x0(
    run_id: int, x0: np.ndarray, hess_invs: dict[int, np.ndarray]
):
    collector = MetricsCollector([BestSoFar()], "bfgs", run_id)
    for label, hess_inv in hess_invs.items():
        run_bfgs(
            x0=x0,
            collector=collector,
            hess_inv=hess_inv,
            identifier=str(label),
        )
    run_bfgs(
        x0=x0,
        collector=collector,
        hess_inv=np.eye(DIMENSIONS),
        identifier="identity",
    )
    return collector.as_dataframe()


def run_bfgs(
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
    cmaes_dfs = [pair[0] for pair in rv]
    bfgs_dfs = [pair[1] for pair in rv]
    bfgs_raw_dfs = [pair[2] for pair in rv]

    cmaes_concatenated = pd.concat(cmaes_dfs)
    new_idx = pd.MultiIndex.from_arrays(
        [
            cmaes_concatenated["run_id"],
            cmaes_concatenated["iteration"],
        ],
        names=["run_id", "iteration"],
    )

    reindexed = cmaes_concatenated.drop(columns=["iteration", "run_id"]).set_index(
        new_idx
    )
    reindexed.to_parquet(
        RESULT_DIR / f"cmaes_d{DIMENSIONS}.parquet",
        index=True,
        compression="brotli",
    )

    aggregate_dataframes(bfgs_dfs).to_parquet(
        RESULT_DIR / f"bfgs_d{DIMENSIONS}.parquet", index=True, compression="brotli"
    )

    bfgs_raw_concat = pd.concat(bfgs_raw_dfs)
    bfgs_raw_concat.set_index(
        pd.MultiIndex.from_arrays(
            [
                bfgs_raw_concat["run_id"],
                bfgs_raw_concat["x0_idx"],
                bfgs_raw_concat.index.values,
            ],
            names=["run_id", "x0_idx", "num_evaluations"],
        )
    ).to_parquet(
        RESULT_DIR / f"bfgs_d{DIMENSIONS}_raw.parquet",
        index=True,
        compression="brotli",
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL)

    os.makedirs(RESULT_DIR, exist_ok=True)

    main()
