import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame
from sympy import prime

from lib.funs import get_function_by_name
from lib.metrics import BestSoFar
from lib.metrics_collector import MetricsCollector
from lib.optimizers.bfgs import BFGS
from lib.optimizers.hybrids import MultiCMABFGS, MultiCMALBFGSB
from lib.optimizers.hybrids.goldencmaes import GoldenCMAES
from lib.serde import aggregate_dataframes
from lib.stopping import BFGSEarlyStopping, CMAESEarlyStopping
from lib.util import EvalCounter

LOG_LEVEL = "ERROR"
DEBUG = os.getenv("DEBUG", False)
PLOT_SUFFIX = "(restart bez uwarunkowania)"
MULTI_CLASS = MultiCMALBFGSB

BOUNDS = 100

if DEBUG:
    DIMENSIONS = 10
    NUM_RUNS = 10
    EXACT_RUN = 6
    OBJECTIVE_NAME = "CEC3"
    SWITCH_AFTER_ITERATIONS = [1, 2, 7, 19, 25, 50, 100, 187, 250, 500, 750]
else:
    DIMENSIONS = int(os.environ["DIMENSIONS"])
    NUM_RUNS = int(os.environ["N_RUNS"])
    OBJECTIVE_NAME = os.environ["OBJECTIVE"]
    SWITCH_AFTER_ITERATIONS = list(map(int, os.environ["SWITCH_AFTER"].split("-")))

MAXEVALS = 10_000 * DIMENSIONS
POPULATION_SIZE = 4 * DIMENSIONS


RESULT_DIR = Path(__file__).parent / "results" / MULTI_CLASS.__name__.lower()
DATA_DIR = RESULT_DIR / "data"
PLOT_EXPORT_DIR = RESULT_DIR / "plots"

colors = plt.cm.tab20.colors  # pyright: ignore[reportAttributeAccessIssue]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)


def run_multi_hybrid(
    klass: type[MultiCMABFGS | MultiCMALBFGSB],
    objective: Callable[[np.ndarray], float | np.ndarray],
    x: np.ndarray,
    seed: int,
    idx: int,
):
    class_to_collection_method = {
        MultiCMABFGS: "cmabfgs",
        MultiCMALBFGSB: "cmabfgs",
        GoldenCMAES: "goldencmaes",
    }
    collection_method = class_to_collection_method[klass]

    counter = EvalCounter(objective, bounds=(-BOUNDS, BOUNDS))
    metrics = [
        BestSoFar(),
    ]
    callback = MetricsCollector(metrics, collection_method, idx)
    optimizer = klass(
        x,
        SWITCH_AFTER_ITERATIONS,
        seed,
        counter,
        POPULATION_SIZE,
        callback,
        CMAESEarlyStopping(MAXEVALS, tolfun=1e-9),
        bounds=(-BOUNDS, BOUNDS),
        restart_cmaes=True,
    )
    optimizer.optimize()
    return callback.as_dataframe()


def run_bfgs(
    objective: Callable[[np.ndarray], float | np.ndarray],
    x: np.ndarray,
    seed: int,
    idx: int,
):
    counter = EvalCounter(objective)
    metrics = [BestSoFar()]
    callback = MetricsCollector(metrics, "bfgs", idx)
    bfgs = BFGS(
        x,
        fun=counter,
        callback=callback,
        stopper=BFGSEarlyStopping(MAXEVALS),
        bounds=(-BOUNDS, BOUNDS),
        identifier="bfgs",
    )
    bfgs.optimize()
    logger.info(f"{idx}: done with BFGS")
    return callback.as_dataframe()


def single_run(idx: int) -> DataFrame:
    try:
        objective, optimum = cast(
            tuple[Callable, float],
            get_function_by_name(OBJECTIVE_NAME, DIMENSIONS, with_optimum=True),
        )
        seed: int = prime(idx)  # pyright: ignore[reportAssignmentType]
        rng = np.random.default_rng(seed)
        x = cast(
            np.ndarray,  # pyright: ignore[reportArgumentType]
            (rng.random(DIMENSIONS) - 0.5) * 2 * BOUNDS,  # pyright: ignore[reportArgumentType]
        )
        multi_optimizer = run_multi_hybrid(MULTI_CLASS, objective, x, seed, idx)
        bfgs = run_bfgs(objective, x, seed, idx).drop(columns=["run_id"])
        return multi_optimizer.join(bfgs, how="outer")
    except Exception as e:
        logger.error(f"Error in run {idx}: {e}")
        raise e


def visualize_results(
    df: DataFrame,
    save_to: Path = PLOT_EXPORT_DIR,
    dimensions=DIMENSIONS,
    switch_after_iterations=SWITCH_AFTER_ITERATIONS,
    function_name: str | None = None,
    suffix: str = "",
):
    population_size = 4 * dimensions
    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    df["best_vanilla_cmaes"].dropna().plot(ax=ax)
    df["best_bfgs"].dropna().plot(ax=ax)

    for i, val in enumerate(switch_after_iterations):
        df[f"best_{val}"].dropna().plot(
            ax=ax, label=f"{val} it/{val * population_size} eval)"
        )

    ymin, ymax = plt.ylim()
    plt.vlines(
        x=np.array(switch_after_iterations) * population_size,
        ymin=ymin,
        ymax=ymax,
        colors="r",
        linestyles="dashed",
        zorder=1.99,
    )

    plt.yscale("log")
    plt.xscale("log")
    plt.grid(which="both")
    plt.xlabel("Liczba ewaluacji")
    plt.ylabel("best so far")
    obj_name = function_name if function_name is not None else OBJECTIVE_NAME
    plt.title(
        (
            f"BFGS vs CMAES vs {MULTI_CLASS.__name__}, funkcja {obj_name}, {dimensions} wymiarów "
            + suffix
        ).strip()
    )
    plt.legend()
    plt.savefig(save_to / f"{obj_name}_{dimensions}.png", dpi=300)


def main():
    with mp.Pool(mp.cpu_count()) as pool:
        run_indices = (
            range(1, NUM_RUNS + 1)
            if not DEBUG or EXACT_RUN is None
            else range(EXACT_RUN, EXACT_RUN + 1)
        )
        rv = pool.map(single_run, run_indices)

    concatenated = pd.concat(rv)
    concatenated["run_id"] = concatenated["run_id"].astype("Int64")
    concatenated.to_parquet(
        DATA_DIR / f"{OBJECTIVE_NAME}_{DIMENSIONS}.parquet",
        index=True,
        compression="brotli",
    )

    agg = aggregate_dataframes(rv)
    visualize_results(agg, suffix=PLOT_SUFFIX)  # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL)

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_EXPORT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    main()
