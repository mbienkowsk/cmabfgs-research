import glob
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from loguru import logger
from omegaconf import OmegaConf

from lib.bound_handling import BoundEnforcement
from lib.funs import ObjectiveFunction, elliptic_hess_inv_for_dim, get_function_by_name
from lib.metrics import BestSoFar
from lib.metrics_collector import MetricsCollector
from lib.optimizers import BFGS
from lib.random import IndividualGenerator
from lib.serde import interpolate_and_stack
from lib.stopping import BFGSEarlyStopping
from lib.util import EvalCounter, compress_and_save, run_indices_pgbar, summarize_data


@dataclass
class BScaleComparisonExperimentConfig:
    mode: Literal["run", "postprocess"]
    num_runs: int
    dimensions: int
    bounds: tuple[float, float]
    scaling: float | Literal["adaptive"]
    noise: float
    bound_enforcement: Literal["ignore_solutions", "additive_penalty"]
    probe_step_size: float = 1e-3

    @property
    def result_dir(self):
        return (
            Path(__file__).parent
            / "results"
            / f"d{self.dimensions}"
            / f"bounds_{int(self.bounds[1])}"
            / f"bound_enforcement_{self.bound_enforcement}"
            / f"noise_{self.noise}"
            / f"probe_{self.probe_step_size}"
            / f"scaling_{self.scaling}"
        )

    @classmethod
    def from_omegaconf(cls, cfg: OmegaConf):
        (
            num_runs,
            dimensions,
            bounds,
            scaling,
            noise,
            bound_enforcement,
            probe_step_size,
        ) = (
            cfg["num_runs"],  # pyright: ignore[reportIndexIssue]
            cfg["dimensions"],  # pyright: ignore[reportIndexIssue]
            cfg["bounds"],  # pyright: ignore[reportIndexIssue]
            cfg["scaling"],  # pyright: ignore[reportIndexIssue]
            cfg["noise"],  # pyright: ignore[reportIndexIssue]
            cfg["bound_enforcement"],  # pyright: ignore[reportIndexIssue]
            cfg["probe_step_size"],  # pyright: ignore[reportIndexIssue]
        )
        return cls(
            mode=cfg["mode"],  # pyright: ignore[reportIndexIssue]
            num_runs=num_runs,
            dimensions=dimensions,
            bounds=(-bounds, bounds),
            scaling=scaling,
            noise=noise,
            bound_enforcement=bound_enforcement,
            probe_step_size=probe_step_size,
        )


def central_diff_jac(fun, x):
    x = np.asarray(x, dtype=float)
    n = x.size

    rel_step = np.sqrt(np.finfo(float).eps)
    h = rel_step * np.maximum(1.0, np.abs(x))

    grad = np.empty(n, dtype=float)

    for i in range(n):
        x_f = x.copy()
        x_b = x.copy()

        x_f[i] += h[i]
        x_b[i] -= h[i]

        f_f = fun(x_f)
        f_b = fun(x_b)

        grad[i] = (f_f - f_b) / (2.0 * h[i])

    return grad


def scale_hess_by_probing(
    fun: ObjectiveFunction,
    x0: np.ndarray,
    B: np.ndarray,
    probe_step_size=1e-3,
):
    g0 = central_diff_jac(
        fun,
        x0,
    )
    sd = B.dot(g0)
    sd_norm = sd / np.linalg.norm(sd)

    x_probe = x0 - probe_step_size * sd_norm
    g_probe = central_diff_jac(fun, x_probe)

    p = x_probe - x0
    y = g_probe - g0

    b = np.dot(y, p)
    a = np.dot(y, np.dot(B, y))
    # alpha = y.T @ p / (y.T @ B @ y)

    alpha = b / a
    if alpha < 0:
        raise ValueError("Negative alpha")
    return alpha * B


@dataclass
class BScaleComparisonExperiment:
    cfg: BScaleComparisonExperimentConfig

    def run_subprocess(self, run_id: int):
        rng = IndividualGenerator(run_id, self.cfg.bounds, self.cfg.dimensions)
        collector = MetricsCollector([BestSoFar()], run_id)
        fn = get_function_by_name("Elliptic")
        match self.cfg.bound_enforcement:
            case "ignore_solutions":
                bound_enforcement_method = BoundEnforcement.IGNORE_SOLUTIONS
            case "additive_penalty":
                bound_enforcement_method = BoundEnforcement.ADDITIVE_PENALTY
            case _:
                raise ValueError(
                    f"Unknown bound enforcement method: {self.cfg.bound_enforcement}"
                )
        counter = EvalCounter(
            fn,  # pyright: ignore[reportArgumentType]
            bounds=self.cfg.bounds,
            bound_enforcement_method=bound_enforcement_method,
        )
        hess_inv = elliptic_hess_inv_for_dim(self.cfg.dimensions)
        x0 = rng.get_individual()

        if self.cfg.scaling == "adaptive":
            scaled_h_inv = scale_hess_by_probing(
                counter, x0, hess_inv, probe_step_size=self.cfg.probe_step_size
            )
        else:
            assert isinstance(self.cfg.scaling, float)
            scaled_h_inv = self.cfg.scaling * hess_inv

        bfgs = BFGS(
            x0,
            counter,
            collector,
            BFGSEarlyStopping(None),
            self.cfg.bounds,
            scaled_h_inv,
        )
        bfgs.optimize()
        return collector.as_dataframe()

    def run(self):
        dfs = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.run_subprocess)(run_id)
            for run_id in run_indices_pgbar(self.cfg.num_runs)
        )
        raw = pd.concat(dfs)  # pyright: ignore[reportCallIssue, reportArgumentType]
        raw["scaling"] = self.cfg.scaling
        raw["noise"] = self.cfg.noise
        compress_and_save(raw, self.cfg.result_dir / "raw.parquet")
        summarize_data(raw)
        print("== Best for each run ===")
        print(raw.groupby("run_id")["best"].min())


def postprocess_and_visualize(config: BScaleComparisonExperimentConfig):
    pattern = str(
        Path(__file__).parent
        / "results"
        / f"d{config.dimensions}"
        / f"bounds_{int(config.bounds[1])}"
        / f"bound_enforcement_{config.bound_enforcement}"
        / f"noise_{config.noise}"
        / f"probe_{config.probe_step_size}"
        / "*"
        / "raw.parquet"
    )
    files = glob.glob(pattern)
    dfs: list[pd.DataFrame] = []
    for file in files:
        df = pd.read_parquet(file)
        scaling = df["scaling"].iloc[0]
        df_interp = interpolate_and_stack(
            [frame for _, frame in df.drop(columns=["scaling"]).groupby("run_id")]
        ).assign(scaling=scaling)
        dfs.append(df_interp)

    df_all = pd.concat(dfs).reset_index()
    plt.figure(figsize=(8, 5))

    df_all["scaling"] = df_all["scaling"].astype("category")

    sns.lineplot(
        data=df_all,
        x="num_evaluations",
        y="best",
        hue="scaling",
        estimator="mean",
        errorbar=(
            "pi",
            50,
        ),
    )

    plt.yscale("log")
    plt.title(
        f"Krzywe zbieżności w zależności od skalowania; d={config.dimensions}, ograniczenia={config.bounds}\n egzekwowanie: {'kara' if config.bound_enforcement == 'additive_penalty' else 'brak'}, szum={config.noise}"
    )
    plt.tight_layout()
    plt.savefig(config.result_dir.parent / "plot.png", dpi=300)
    plt.show()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: OmegaConf):
    config = BScaleComparisonExperimentConfig.from_omegaconf(cfg)
    config.result_dir.mkdir(parents=True, exist_ok=True)

    if config.mode == "run":
        exp = BScaleComparisonExperiment(config)
        exp.run()
    else:
        postprocess_and_visualize(config)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    main()
