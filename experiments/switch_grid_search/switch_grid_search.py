import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from pandas import DataFrame
from sympy import prime

from lib.callbacks import MetricsCollector
from lib.funs import get_function_by_name
from lib.metrics import BestSoFar
from lib.optimizers.bfgs import BFGS
from lib.optimizers.multicmabfgs import MultiCMABFGS
from lib.serde import aggregate_dataframes
from lib.stopping import BFBGSEarlyStopping, CMAESEarlyStopping
from lib.util import EvalCounter

BOUNDS = 100
DIMENSIONS = int(os.environ["DIMENSIONS"])
NUM_RUNS = int(os.environ["N_RUNS"])
OBJECTIVE_NAME = os.environ["OBJECTIVE"]
SWITCH_AFTER_ITERATIONS = list(map(int, os.environ["SWITCH_AFTER"].split("-")))
# DIMENSIONS = 100
# NUM_RUNS = 5
# OBJECTIVE_NAME = "CEC21"
# SWITCH_AFTER_ITERATIONS = [750]
#
OBJECTIVE, OPTIMUM = cast(
    tuple[Callable, float],
    get_function_by_name(OBJECTIVE_NAME, DIMENSIONS, with_optimum=True),
)
MAXEVALS = 10_000 * DIMENSIONS
POPULATION_SIZE = 4 * DIMENSIONS


RESULT_DIR = Path(__file__).parent / "results"
DATA_DIR = RESULT_DIR / "data"
PLOT_EXPORT_DIR = RESULT_DIR / "plots"

colors = plt.cm.tab20.colors  # pyright: ignore[reportAttributeAccessIssue]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)


def run_multicmabfgs(x: np.ndarray, seed: int, idx: int):
    counter = EvalCounter(OBJECTIVE)
    metrics = [
        BestSoFar(OPTIMUM),
    ]
    callback = MetricsCollector(metrics, "cmabfgs")
    optimizer = MultiCMABFGS(
        x,
        SWITCH_AFTER_ITERATIONS,
        seed,
        counter,
        POPULATION_SIZE,
        callback,
        CMAESEarlyStopping(MAXEVALS, tolfun=1e-9),
        bounds=(-BOUNDS, BOUNDS),
    )
    optimizer.optimize()
    return callback.as_dataframe()


def run_bfgs(x: np.ndarray, seed: int, idx: int):
    counter = EvalCounter(OBJECTIVE)
    metrics = [BestSoFar(OPTIMUM)]
    callback = MetricsCollector(metrics, "bfgs")
    bfgs = BFGS(
        x,
        seed=seed,
        fun=counter,
        callback=callback,
        stopper=BFBGSEarlyStopping(MAXEVALS),
        bounds=(-BOUNDS, BOUNDS),
    )
    bfgs.optimize()
    logger.info(f"{idx}: done with BFGS")
    return callback.as_dataframe()


def single_run(idx: int) -> tuple[DataFrame, DataFrame]:
    try:
        seed: int = prime(idx)  # pyright: ignore[reportAssignmentType]
        rng = np.random.default_rng(seed)
        x = cast(
            np.ndarray,  # pyright: ignore[reportArgumentType]
            (rng.random(DIMENSIONS) - 0.5) * 2 * BOUNDS,  # pyright: ignore[reportArgumentType]
        )
        cmabfgs = run_multicmabfgs(x, seed, idx)
        bfgs = run_bfgs(x, seed, idx)
        return cmabfgs, bfgs
    except Exception as e:
        logger.error(f"Error in run {idx}: {e}")
        raise e


def visualize_results(
    bfgs: DataFrame,
    cmabfgs: DataFrame,
    save_to: Path = PLOT_EXPORT_DIR,
    dimensions=DIMENSIONS,
    switch_after_iterations=SWITCH_AFTER_ITERATIONS,
    function_name: str | None = None,
):
    population_size = 4 * dimensions
    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    if len(bfgs) == 1:
        plt.plot(bfgs.index, bfgs["best"], label="BFGS", marker="o")
    else:
        plt.plot(bfgs.index[1:], bfgs["best"][1:], label="BFGS")

    cmabfgs["best_vanilla_cmaes"].dropna().plot(ax=ax)

    for i, val in enumerate(switch_after_iterations):
        cmabfgs[f"best_{val}"].dropna().plot(
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
    plt.title(f"BFGS vs CMAES vs CMABFGS, funkcja {obj_name}, {dimensions} wymiarów")
    plt.legend()
    plt.savefig(save_to / f"{obj_name}_{dimensions}.png", dpi=300)


def main():
    with mp.Pool(mp.cpu_count()) as pool:
        rv = pool.map(single_run, range(1, NUM_RUNS + 1))

    cmabfgs_agg = aggregate_dataframes([val[0] for val in rv])
    bfgs_agg = aggregate_dataframes([val[1] for val in rv])

    # Prefix columns
    bfgs_prefixed = bfgs_agg.add_prefix("bfgs_")
    # Concatenate along columns
    combined = bfgs_prefixed.join(cmabfgs_agg, how="outer")
    # Save to CSV
    combined.to_csv(
        DATA_DIR / f"{OBJECTIVE_NAME}_{DIMENSIONS}_combined.csv",
        index_label="num_evaluations",
    )
    visualize_results(bfgs_agg, cmabfgs_agg)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_EXPORT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    main()
