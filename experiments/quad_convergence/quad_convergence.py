import multiprocessing as mp
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from lib.funs import elliptic_hess_for_dim, get_function_by_name
from lib.metrics import BestSoFar, BestXSoFar, CovarianceMatrix, Mean
from lib.metrics_collector import MetricsCollector
from lib.optimizers.bfgs import BFGS
from lib.optimizers.cmaes import CMAES
from lib.serde import aggregate_dataframes
from lib.stopping import BFGSEarlyStopping, CMAESEarlyStopping
from lib.util import (
    EvalCounter,
    assert_all_non_increasing,
    assert_non_increasing,
    get_x0_and_seed_for_run_id,
)

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
    CMAES_COLLECTION_INTERVAL = 20

else:
    DIMENSIONS = int(os.environ["DIMENSIONS"])
    NUM_RUNS = int(os.environ["N_RUNS"])
    CMAES_COLLECTION_INTERVAL = int(os.environ["CMAES_COLLECTION_INTERVAL"])

MAXEVALS = 10_000 * DIMENSIONS
POPULATION_SIZE = 4 * DIMENSIONS
RESULT_DIR = Path(__file__).parent / "results" / f"d{DIMENSIONS}"
AGG_DIR = RESULT_DIR / "agg"
RAW_DIR = RESULT_DIR / "raw"
X0_GENERATOR_SEED = 7

GROUND_TRUTH_INV_HESS = np.invert(elliptic_hess_for_dim(DIMENSIONS))


def run_cmaes(run_id: int):
    objective = get_function_by_name(OBJECTIVE_NAME, DIMENSIONS)
    counter = EvalCounter(objective, bounds=(-BOUNDS, BOUNDS))  # pyright: ignore[reportArgumentType]
    x, seed = get_x0_and_seed_for_run_id(run_id, DIMENSIONS, BOUNDS)
    collector = MetricsCollector(
        [
            CovarianceMatrix(),
            BestXSoFar(),
            Mean(),
        ],
        run_id,
        every_n_calls=CMAES_COLLECTION_INTERVAL,
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

    return df.assign(iteration=df.index // POPULATION_SIZE)


def make_symmetrical(mat: np.ndarray):
    return mat * 0.5 + mat.T * 0.5


def normalize(mat: np.ndarray):
    return mat / np.linalg.norm(mat)


def single_run(
    run_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns 5 dfs:
    * cmaes run for this id with cov mat and mean info
    * bfgs aggregated without inheriting x0
    * raw concatenated bfgs without inheriting x0
    * bfgs aggregated with inheriting x0
    * raw concatenated bfgs with inheriting x0
    """
    cmaes_df = run_cmaes(run_id)
    num_x0s = len(cmaes_df)
    rng = np.random.default_rng(X0_GENERATOR_SEED)
    x0s = rng.uniform(low=-BOUNDS, high=BOUNDS, size=(num_x0s, DIMENSIONS)).reshape(
        (num_x0s, DIMENSIONS)
    )
    return (
        cmaes_df,  # pyright: ignore[reportReturnType]
        *run_bfgs_with_predefined_x0s(run_id, cmaes_df, x0s),  # pyright: ignore[reportReturnType]
        *run_bfgs_with_inherited_means(run_id, cmaes_df),
    )


def run_bfgs_with_predefined_x0s(run_id: int, df: pd.DataFrame, x0s: np.ndarray):
    by_iter = df.set_index("iteration")
    subrun_dfs: list[pd.DataFrame] = []
    for i in range(len(x0s)):
        x0 = x0s[i]
        collector = MetricsCollector([BestSoFar()], run_id)
        for iters, row in by_iter.iterrows():
            run_bfgs(
                x0=x0,
                collector=collector,
                hess_inv=make_symmetrical(row["cov_mat"]),  # pyright: ignore[reportArgumentType]
                identifier=f"{iters}_random_x0",
            )
            run_bfgs(
                x0=x0,
                collector=collector,
                hess_inv=make_symmetrical(normalize(row["cov_mat"])),  # pyright: ignore[reportArgumentType]
                identifier=f"{iters}_inherited_x0_normalized",
            )
        run_bfgs(
            x0=x0,
            collector=collector,
            hess_inv=np.eye(DIMENSIONS),
            identifier="identity",
        )
        run_bfgs(
            x0=x0,
            collector=collector,
            hess_inv=deepcopy(GROUND_TRUTH_INV_HESS),
            identifier="identity",
        )
        subrun_dfs.append(collector.as_dataframe().assign(subrun_id=i))

    raw_result = pd.concat(subrun_dfs)
    assert_all_non_increasing(subrun_dfs)
    agg_result = aggregate_dataframes(subrun_dfs, "subrun_id")
    try:
        assert_non_increasing(agg_result)
    except AssertionError:
        print(agg_result)
        print(agg_result.dtypes)
        print(agg_result.diff())
        agg_result.to_parquet("debug.parquet")
    return agg_result, raw_result


def run_bfgs_with_inherited_means(run_id: int, df: pd.DataFrame):
    by_iter = df.set_index("iteration")
    subrun_dfs: list[pd.DataFrame] = []
    for subrun_id in range(NUM_RUNS):
        collector = MetricsCollector([BestSoFar()], run_id)
        for iters, row in by_iter.iterrows():
            run_bfgs(
                x0=row["mean"],  # pyright: ignore[reportArgumentType]
                collector=collector,
                hess_inv=make_symmetrical(row["cov_mat"]),  # pyright: ignore[reportArgumentType]
                identifier=f"{iters}_inherited_x0",
            )
            run_bfgs(
                x0=row["mean"],  # pyright: ignore[reportArgumentType]
                collector=collector,
                hess_inv=make_symmetrical(normalize(row["cov_mat"])),  # pyright: ignore[reportArgumentType]
                identifier=f"{iters}_inherited_x0_normalized",
            )
        run_bfgs(
            x0=row["mean"],  # pyright: ignore[reportArgumentType]
            collector=collector,
            hess_inv=np.eye(DIMENSIONS),
            identifier="identity_inherited_x0",
        )
        run_bfgs(
            x0=row["mean"],  # pyright: ignore[reportArgumentType]
            collector=collector,
            # TODO: scale down?
            hess_inv=deepcopy(GROUND_TRUTH_INV_HESS),
            identifier="identity",
        )
        subrun_dfs.append(collector.as_dataframe().assign(subrun_id=subrun_id))

    raw_result = pd.concat(subrun_dfs)
    assert_all_non_increasing(subrun_dfs)
    agg_result = aggregate_dataframes(subrun_dfs, "subrun_id")
    assert_non_increasing(agg_result)
    return agg_result, raw_result


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


def reindex_and_save_raw_data(dfs: list[pd.DataFrame], fname: str):
    raw_concat = pd.concat(dfs)
    reindexed = raw_concat.set_index(
        pd.MultiIndex.from_arrays(
            [
                raw_concat["run_id"],
                raw_concat["subrun_id"],
                raw_concat.index.values,
            ],
            names=["run_id", "subrun_id", "num_evaluations"],
        )
    )
    reindexed.to_parquet(
        RAW_DIR / fname,
        index=True,
        compression="brotli",
    )


def aggregate_and_save_agg_data(dfs: list[pd.DataFrame], fname: str):
    assert_all_non_increasing(dfs)
    agg = aggregate_dataframes(dfs)
    assert_non_increasing(agg)
    agg.to_parquet(
        AGG_DIR / fname,
        index=True,
        compression="brotli",
    )


def main():
    with mp.Pool(mp.cpu_count()) as pool:
        run_indices = (
            range(1, NUM_RUNS + 1)
            if not DEBUG or EXACT_RUN is None
            else range(EXACT_RUN, EXACT_RUN + 1)
        )
        rv = pool.map(single_run, run_indices)
    cmaes_dfs = [pair[0] for pair in rv]
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
    # has to be flattened before compressing
    reindexed["cov_mat"] = reindexed["cov_mat"].apply(np.ravel)
    reindexed.to_parquet(
        RAW_DIR / "cmaes.parquet",
        index=True,
        compression="brotli",
    )

    # aggregated with predefined x0s
    aggregate_and_save_agg_data([v[1] for v in rv], "bfgs.parquet")
    # aggregated inheriting means
    aggregate_and_save_agg_data([v[3] for v in rv], "bfgs_inherited_x0.parquet")
    reindex_and_save_raw_data([v[2] for v in rv], "bfgs.parquet")
    reindex_and_save_raw_data([v[4] for v in rv], "bfgs_inherited_x0.parquet")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL)

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(AGG_DIR, exist_ok=True)

    main()
