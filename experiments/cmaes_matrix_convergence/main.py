import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sympy import prime

from lib.callbacks import MetricsCollector
from lib.funs import get_function_by_name
from lib.metrics import (BestSoFar, CovarianceMatrixConditionNumber,
                         CovarianceMatrixDifferenceNorm)
from lib.optimizers.cmaes import CMAES
from lib.serde import load_results_from_directory
from lib.stopping import CMAESEarlyStopping
from lib.util import (EvalCounter, extract_dim_from_path,
                      extract_objective_from_path)

RESULT_DIR = Path(__file__).parent / "results"


BOUNDS = 100
DIMENSIONS = int(os.environ["DIMENSIONS"])
NUM_RUNS = int(os.environ["N_RUNS"])
OBJECTIVE_NAME = os.environ["OBJECTIVE"]
# DIMENSIONS = 5
# NUM_RUNS = 3
# OBJECTIVE_NAME = "CEC10"
OBJECTIVE = get_function_by_name(OBJECTIVE_NAME, DIMENSIONS)
MAXEVALS = 4000 * DIMENSIONS
POPULATION_SIZE = 4 * DIMENSIONS


PLOT_EXPORT_DIR = RESULT_DIR / "plots"
RESULT_DIR = Path(__file__).parent / f"results/fun_{OBJECTIVE_NAME}_dim_{DIMENSIONS}"


def run_vanilla(x: np.ndarray, seed: int, idx: int):
    counter = EvalCounter(OBJECTIVE)
    metrics = [
        BestSoFar(),
        CovarianceMatrixConditionNumber(),
        CovarianceMatrixDifferenceNorm(),
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

    callback.export_to_csv(
        RESULT_DIR / f"{idx}.csv",
    )
    logger.info(f"{idx}: done")


def visualize_results(result_path: Path):
    data = load_results_from_directory(result_path)
    dim = extract_dim_from_path(result_path)
    objective = extract_objective_from_path(result_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    ax1.plot(data.index, data.iloc[:, 1], label=data.columns[1])
    ax1.set_title("Współczynnik uwarunkowania C vs liczba ewaluacji")
    ax1.set_ylabel("cond(C)")
    ax1.legend()

    ax2.plot(data.index, data.iloc[:, 2], label=data.columns[2])
    ax2.set_title("Kwadrat zmiany C vs liczba ewaluacji")

    for ax in (ax1, ax2):
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("liczba ewaluacji")
        ax.legend()

    fig.suptitle(f"Funkcja {objective} w {dim} wymiarach")
    fig.tight_layout()
    fig.savefig(result_path / f"{objective}_{dim}.png", dpi=300)


def single_run(idx: int):
    seed: int = prime(idx)  # pyright: ignore[reportAssignmentType]
    rng = np.random.default_rng(seed)
    x = cast(
        np.ndarray,  # pyright: ignore[reportArgumentType]
        (rng.random(DIMENSIONS) - 0.5)
        * 2
        * BOUNDS,  # pyright: ignore[reportArgumentType]
    )
    run_vanilla(x, seed, idx)


def main():
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(single_run, range(1, NUM_RUNS + 1))


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)

    os.makedirs(RESULT_DIR)

    main()
    visualize_results(PLOT_EXPORT_DIR)
