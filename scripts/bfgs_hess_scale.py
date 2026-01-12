import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from loguru import logger

from lib.funs import elliptic_grad, elliptic_hess_for_dim, get_function_by_name
from lib.metrics import BestSoFar
from lib.metrics_collector import MetricsCollector
from lib.optimizers.bfgs import BFGS
from lib.stopping import BFGSEarlyStopping
from lib.util import EvalCounter, evaluation_budget, get_x0_and_seed_for_run_id

DIM = 100
MAXEVALS = evaluation_budget(DIM)
RESULT_PATH = Path(__file__).parent / "bfgs_hess_scale.parquet"
JAC = elliptic_grad
KILL_OUTSIDE_BOUNDS = True
BFGS_BOUNDS = (-100, 100) if KILL_OUTSIDE_BOUNDS else (-1e9, 1e9)
RUN_ID = 1


def get_scaled_hess_inv(fun, jac, x0, M, epsilon=1e-5):
    g0 = jac(x0)
    p = -M.dot(g0)
    p_norm = p / np.linalg.norm(p)

    x_probe = x0 + epsilon * p_norm
    g_probe = jac(x_probe)

    s = x_probe - x0
    y = g_probe - g0

    denom = np.dot(y, np.dot(M, y))
    num = np.dot(y, s)
    # alpha = y.T @ s / (y.T @ M @ y)

    alpha = num / denom
    return alpha * M


def normalize(mat: np.ndarray):
    return mat / np.linalg.norm(mat)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    objective = get_function_by_name("Elliptic", dim=DIM)
    xstart, seed = get_x0_and_seed_for_run_id(RUN_ID, DIM, 100)
    SCALING_FACTORS = (1e-9, 1e-6, 1e-3, 1, 1.01, 1e3, 1e6, 1e9)
    hess = elliptic_hess_for_dim(DIM)
    inv_hess = np.linalg.inv(hess)
    print(f"inv_hess norm: {np.linalg.norm(inv_hess)}")

    metrics = (BestSoFar(),)
    callback = MetricsCollector(deepcopy(metrics), RUN_ID)

    for factor in SCALING_FACTORS:
        BFGS(
            xstart,
            EvalCounter(objective, bounds=(-100, 100)),  # pyright: ignore[reportArgumentType]
            callback,
            BFGSEarlyStopping(MAXEVALS),
            bounds=BFGS_BOUNDS,
            identifier=f"inv_hess * {factor}",
            hess_inv0=inv_hess * factor,
        ).optimize()

        BFGS(
            xstart,
            EvalCounter(objective, bounds=(-100, 100)),  # pyright: ignore[reportArgumentType]
            callback,
            BFGSEarlyStopping(MAXEVALS),
            bounds=BFGS_BOUNDS,
            identifier=f"inv_hess * {factor}, then normalized",
            hess_inv0=normalize(inv_hess * factor),
        ).optimize()

        scaled_by_probing = get_scaled_hess_inv(
            objective,
            JAC,
            xstart,
            inv_hess * factor,
        )
        BFGS(
            xstart,
            EvalCounter(objective, bounds=(-100, 100)),  # pyright: ignore[reportArgumentType]
            callback,
            BFGSEarlyStopping(MAXEVALS),
            bounds=BFGS_BOUNDS,
            identifier=f"inv_hess * {factor}, norm_probe",
            hess_inv0=normalize(inv_hess * factor),
        ).optimize()

    BFGS(
        xstart,
        EvalCounter(objective, bounds=(-100, 100)),  # pyright: ignore[reportArgumentType]
        callback,
        BFGSEarlyStopping(MAXEVALS),
        bounds=BFGS_BOUNDS,
        identifier="diagonal (no precond.)",
    ).optimize()

    callback.as_dataframe().to_parquet(RESULT_PATH)
