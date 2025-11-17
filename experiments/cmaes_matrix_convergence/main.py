import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sympy import prime

from lib.funs import get_function_by_name
from lib.metrics import (
    BestSoFar,
    CovarianceMatrixConditionNumber,
    CovarianceMatrixDifferenceNorm,
)
from lib.metrics_collector import MetricsCollector
from lib.optimizers.cmaes import CMAES
from lib.serde import load_results_from_directory
from lib.stopping import CMAESEarlyStopping
from lib.util import EvalCounter, extract_dim_from_path, extract_objective_from_path

RESULT_DIR = Path(__file__).parent / "results"


BOUNDS = 100
DIMENSIONS = int(os.environ["DIMENSIONS"])
NUM_RUNS = int(os.environ["N_RUNS"])
OBJECTIVE_NAME = os.environ["OBJECTIVE"]
# DIMENSIONS = 5
# NUM_RUNS = 3
# OBJECTIVE_NAME = "CEC10"
OBJECTIVE, OPTIMUM = cast(
    tuple[Callable, float],
    get_function_by_name(OBJECTIVE_NAME, DIMENSIONS, with_optimum=True),
)
MAXEVALS = 4000 * DIMENSIONS
POPULATION_SIZE = 4 * DIMENSIONS


PLOT_EXPORT_DIR = RESULT_DIR / "plots"
RESULT_DIR = Path(__file__).parent / f"results/fun_{OBJECTIVE_NAME}_dim_{DIMENSIONS}"


def run_vanilla(x: np.ndarray, seed: int, idx: int):
    counter = EvalCounter(OBJECTIVE)
    metrics = [
        BestSoFar(OPTIMUM),
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


def visualize_results(read_from: Path, save_to: Path):
    data = load_results_from_directory(read_from)
    dim = extract_dim_from_path(read_from)
    objective = extract_objective_from_path(read_from)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6), sharex=True)

    ax1.plot(data.index, data["cov_cond"])
    ax1.set_title("Współczynnik uwarunkowania C vs liczba ewaluacji")
    ax1.set_ylabel("cond(C)")
    ax1.legend()

    ax2.plot(data.index, data["cov_diff_sq"])
    ax2.set_title("Kwadrat zmiany C vs liczba ewaluacji")
    ax2.set_ylabel("cov_diff_sq")

    ax3.plot(data.index, data["best"])
    ax3.set_title(
        "Różnica między wartością f.straty i minimum globalnym w najlepszym punkcie vs liczba ewaluacji"
    )
    ax3.set_ylabel("loss(x_best) - min_global")

    for ax in (ax1, ax2, ax3):
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("liczba ewaluacji")
        ax.legend()
        ax.grid(which="both")

    fig.suptitle(f"Funkcja {objective} w {dim} wymiarach")
    fig.tight_layout()
    fig.savefig(save_to / f"{objective}_{dim}.png", dpi=300)


def visualize_aggregated_results(read_from: Path, save_to: Path):
    """visualize results on a single plot to compare trends"""
    data = load_results_from_directory(read_from)
    dim = extract_dim_from_path(read_from)
    objective = extract_objective_from_path(read_from)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.plot(data.index, data["cov_cond"], label="cov_cond", color="C0")
    ax1.set_ylabel("cov_cond", color="C0")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(data.index, data["cov_diff_sq"], label="cov_diff_sq", color="C1")
    ax2.set_ylabel("cov_diff_sq", color="C1")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="C1")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.plot(data.index, data["best"], label="best", color="C2")
    ax3.set_ylabel("loss(x_best)", color="C2")
    ax3.set_yscale("log")
    ax3.tick_params(axis="y", labelcolor="C2")

    ax1.set_xlabel("liczba ewaluacji f.celu")
    fig.suptitle(
        f"Relacja między wartością najlepszego punktu, wsp. uwarunkowania C i \nkwadratem zmiany C. Funkcja {objective}, {dim} wymiarów"
    )
    ax1.grid(which="both")
    fig.tight_layout()
    fig.legend()
    fig.savefig(save_to / f"{objective}_{dim}_agg.png", dpi=300)


def single_run(idx: int):
    seed: int = prime(idx)  # pyright: ignore[reportAssignmentType]
    rng = np.random.default_rng(seed)
    x = cast(
        np.ndarray,  # pyright: ignore[reportArgumentType]
        (rng.random(DIMENSIONS) - 0.5) * 2 * BOUNDS,  # pyright: ignore[reportArgumentType]
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
    os.makedirs(PLOT_EXPORT_DIR, exist_ok=True)

    main()
    visualize_results(RESULT_DIR, PLOT_EXPORT_DIR)
    visualize_aggregated_results(RESULT_DIR, PLOT_EXPORT_DIR)
