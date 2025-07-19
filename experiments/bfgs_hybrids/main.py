"""Since es.C approximates the inverse of the hessian matrix, test how plugging it into
a gradient method such as BFGS compares to the vanilla CMA-ES and vanilla BFGS/L-BFGS"""

import multiprocessing as mp
import os
import re
import shutil
import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sympy import prime

from lib.callbacks import MetricsCollector
from lib.funs import get_function_by_name
from lib.metrics import BestSoFar
from lib.optimizers import BFGS, CMAES, LBFGS
from lib.optimizers.cmabfgs import CMABFGS
from lib.serde import load_and_interpolate_results
from lib.stopping import CMAESEarlyStopping
from lib.util import EvalCounter

BOUNDS = 100
DIMENSIONS = int(os.environ["DIMENSIONS"])
NUM_RUNS = int(os.environ["N_RUNS"])
SWITCH_AFTER_ITERATIONS = int(os.environ["SWITCH_AFTER"])
OBJECTIVE_NAME = os.environ["OBJECTIVE"]
OBJECTIVE = get_function_by_name(OBJECTIVE_NAME, DIMENSIONS)
MAXEVALS = 4000 * DIMENSIONS
POPULATION_SIZE = 4 * DIMENSIONS


RESULT_DIR = (
    Path(__file__).parent
    / f"results/fun_{OBJECTIVE_NAME}_dim_{DIMENSIONS}/K_{SWITCH_AFTER_ITERATIONS}"
)
VANILLA_RESULT_DIR = RESULT_DIR / "vanilla"
BFGS_RESULT_DIR = RESULT_DIR / "bfgs"
LBFGS_RESULT_DIR = RESULT_DIR / "lbfgs"
CMABFGS_RESULT_DIR = RESULT_DIR / "cmabfgs"


def run_vanilla(x: np.ndarray, seed: int, idx: int):
    counter = EvalCounter(OBJECTIVE)
    metrics = [BestSoFar()]
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
        VANILLA_RESULT_DIR / f"{idx}.csv",
    )
    logger.info(f"{idx}: done with CMA-ES")


def run_bfgs(x: np.ndarray, seed: int, idx: int):
    counter = EvalCounter(OBJECTIVE)
    metrics = [BestSoFar()]
    callback = MetricsCollector(metrics, "bfgs")
    bfgs = BFGS(x, seed=seed, fun=counter, callback=callback)
    bfgs.optimize()
    callback.export_to_csv(
        BFGS_RESULT_DIR / f"{idx}.csv",
    )
    logger.info(f"{idx}: done with BFGS")


def run_lbfgs(x: np.ndarray, seed: int, idx: int):
    counter = EvalCounter(OBJECTIVE)
    metrics = [BestSoFar()]
    callback = MetricsCollector(metrics, "bfgs")
    bfgs = LBFGS(x, seed=seed, fun=counter, callback=callback)
    bfgs.optimize()
    callback.export_to_csv(
        LBFGS_RESULT_DIR / f"{idx}.csv",
    )
    logger.info(f"{idx}: done with L-BFGS")


def run_cmabfgs(x: np.ndarray, seed: int, idx: int):
    counter = EvalCounter(OBJECTIVE)
    metrics = [BestSoFar()]
    callback = MetricsCollector(metrics, "cmabfgs")
    cmabfgs = CMABFGS(
        x,
        seed=seed,
        fun=counter,
        callback=callback,
        n_cmaes_iterations=SWITCH_AFTER_ITERATIONS,
        popsize=POPULATION_SIZE,
    )
    cmabfgs.optimize()
    callback.export_to_csv(CMABFGS_RESULT_DIR / f"{idx}.csv")
    logger.info(f"{idx}: done with CMABFGS")


def visualize_results(result_path: Path):

    fig = plt.figure()
    postscripts = ["vanilla", "bfgs", "lbfgs", "cmabfgs"]

    label_to_dirs = {
        "vanilla CMA-ES": result_path / "vanilla",
        "vanilla BFGS": result_path / "bfgs",
        "vanilla L-BFGS": result_path / "lbfgs",
        "CMA-ES + BFGS": result_path / "cmabfgs",
    }

    for label, dir in label_to_dirs.items():
        x, y = load_and_interpolate_results(str(dir))
        sns.lineplot(
            x=x,
            y=y,
            label=label,
            ax=fig.gca(),
        )

    dim = extract_dim_from_path(result_path)
    objective = extract_objective_from_path(result_path)
    plt.title(f"Funkcja {objective} w {dim} wymiarach, K = {SWITCH_AFTER_ITERATIONS}")
    plt.xlabel("Liczba ewaluacji f. celu")
    plt.ylabel("Najlepsze znalezione rozwiązanie")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(result_path / f"plot.png")


def extract_dim_from_path(path: Path):
    """Extracts the dimension from a path containing 'DIM_<number>'."""
    match = re.search(r"DIM_(\d+)", str(path).upper())
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract dimension from path: {path}")


def extract_k_from_path(path: Path):
    """Extracts the k-value from a path containing 'K_<number>'."""
    match = re.search(r"K_(\d+)", str(path).upper())
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract dimension from path: {path}")


def extract_objective_from_path(path: Path):
    """Extracts the objective function name from a path containing 'FUN_<name>_'."""
    match = re.search(r"FUN_([^_]+)", str(path).upper())
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract objective from path: {path}")


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
    run_bfgs(x, seed, idx)
    run_lbfgs(x, seed, idx)
    run_cmabfgs(x, seed, idx)


def main():
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(single_run, range(1, NUM_RUNS + 1))


if __name__ == "__main__":

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)

    os.makedirs(RESULT_DIR)
    os.makedirs(VANILLA_RESULT_DIR)
    os.makedirs(BFGS_RESULT_DIR)
    os.makedirs(LBFGS_RESULT_DIR)
    os.makedirs(CMABFGS_RESULT_DIR)

    main()
    visualize_results(RESULT_DIR)
