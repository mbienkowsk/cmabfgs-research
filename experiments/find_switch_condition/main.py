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

from lib.funs import get_function_by_name
from lib.metrics import BestSoFar
from lib.metrics_collector import MetricsCollector
from lib.optimizers.cmaes import CMAES
from lib.optimizers.hybrids import CMABFGS
from lib.serde import aggregate_dataframes
from lib.stopping import CMAESEarlyStopping
from lib.util import EvalCounter

BOUNDS = 100
DIMENSIONS = int(os.environ["DIMENSIONS"])
NUM_RUNS = int(os.environ["N_RUNS"])
OBJECTIVE_NAME = os.environ["OBJECTIVE"]
print(os.environ["SWITCH_AFTER"])
SWITCH_AFTER_VALUES = list(map(int, os.environ["SWITCH_AFTER"].split("-")))

OBJECTIVE, OPTIMUM = cast(
    tuple[Callable, float],
    get_function_by_name(OBJECTIVE_NAME, DIMENSIONS, with_optimum=True),
)
MAXEVALS = 4000 * DIMENSIONS
POPULATION_SIZE = 4 * DIMENSIONS


RESULT_DIR = Path(__file__).parent / "results"
PLOT_EXPORT_DIR = RESULT_DIR / "plots"


def run_vanilla(x: np.ndarray, seed: int, idx: int):
    counter = EvalCounter(OBJECTIVE)
    metrics = [
        BestSoFar(OPTIMUM),
    ]
    callback = MetricsCollector(metrics, "cmaes")
    stopper = CMAESEarlyStopping(max_evals=MAXEVALS, tolfun=1e-9)
    cmaes = CMAES(
        mean=x,
        popsize=POPULATION_SIZE,
        seed=seed,
        fun=counter,
        stopper=stopper,
        callback=callback,
    )
    cmaes.optimize()

    logger.info(f"{idx}: done")
    return callback.as_dataframe()


def run_cmabfgs(x: np.ndarray, seed: int, idx: int, switch_after_objective_calls: int):
    counter = EvalCounter(OBJECTIVE)
    metrics = [BestSoFar(OPTIMUM)]
    callback = MetricsCollector(metrics, "cmabfgs")
    cmabfgs = CMABFGS(
        x,
        seed=seed,
        fun=counter,
        callback=callback,
        n_cmaes_iterations=switch_after_objective_calls // POPULATION_SIZE,
        popsize=POPULATION_SIZE,
    )
    cmabfgs.optimize()
    logger.info(f"{idx}: done with CMABFGS s/a {switch_after_objective_calls}")
    return callback.as_dataframe()


def single_run(idx: int) -> tuple[DataFrame, list[DataFrame]]:
    seed: int = prime(idx)  # pyright: ignore[reportAssignmentType]
    rng = np.random.default_rng(seed)
    x = cast(
        np.ndarray,  # pyright: ignore[reportArgumentType]
        (rng.random(DIMENSIONS) - 0.5) * 2 * BOUNDS,  # pyright: ignore[reportArgumentType]
    )
    vanilla = run_vanilla(x, seed, idx)
    cmabfgs = [
        run_cmabfgs(x, seed, idx, switch_after) for switch_after in SWITCH_AFTER_VALUES
    ]
    return vanilla, cmabfgs


def visualize_results(vanilla: DataFrame, cmabfgs: list[DataFrame], save_to: Path):
    plt.figure(figsize=(9, 6))
    plt.plot(vanilla.index, vanilla["best"], label="Klasyczny CMA-ES")
    for i, val in enumerate(SWITCH_AFTER_VALUES):
        plt.plot(
            cmabfgs[i].index,
            cmabfgs[i]["best"],
            label=f"CMABFGS (switch po {val} ewaluacjach/{val // POPULATION_SIZE} iteracjach)",
        )
    ymin, ymax = plt.ylim()
    plt.vlines(
        x=SWITCH_AFTER_VALUES, ymin=ymin, ymax=ymax, colors="r", linestyles="dashed"
    )
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(which="both")
    plt.xlabel("Liczba ewaluacji")
    plt.legend()
    plt.ylabel("best so far")
    plt.title(f"CMA-ES vs CMABFGS, funkcja {OBJECTIVE_NAME}, {DIMENSIONS} wymiarów")
    plt.savefig(save_to / f"{OBJECTIVE_NAME}_{DIMENSIONS}.png", dpi=300)


def main():
    with mp.Pool(mp.cpu_count()) as pool:
        return_values = pool.map(single_run, range(1, NUM_RUNS + 1))
        vanilla_agg = aggregate_dataframes([res[0] for res in return_values])
        cmabfgs_aggs = [
            aggregate_dataframes([res[1][i] for res in return_values])
            for i in range(len(SWITCH_AFTER_VALUES))
        ]
        # Prefix columns
        vanilla_prefixed = vanilla_agg.add_prefix("vanilla_")
        cmabfgs_prefixed = [
            cmabfgs_aggs[i].add_prefix(f"cmabfgs_{SWITCH_AFTER_VALUES[i]}_")
            for i in range(len(SWITCH_AFTER_VALUES))
        ]
        # Concatenate along columns
        combined = vanilla_prefixed
        for df in cmabfgs_prefixed:
            combined = combined.join(df, how="outer")
        # Save to CSV
        combined.to_csv(RESULT_DIR / f"{OBJECTIVE_NAME}_{DIMENSIONS}_combined.csv")
        visualize_results(vanilla_agg, cmabfgs_aggs, save_to=PLOT_EXPORT_DIR)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_EXPORT_DIR, exist_ok=True)

    main()
