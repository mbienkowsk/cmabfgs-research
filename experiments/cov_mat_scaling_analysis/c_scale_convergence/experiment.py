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

import lib.metrics as m
from lib.bound_handling import BoundEnforcement
from lib.funs import elliptic_hess_inv_for_dim, get_function_by_name
from lib.metrics_collector import MetricsCollector
from lib.optimizers import CMAES
from lib.plotting_util import tex
from lib.random import IndividualGenerator
from lib.stopping import CMAESEarlyStopping
from lib.util import (
    EvalCounter,
    compress_and_save,
    evaluation_budget,
    hansen_cmaes_popsize,
    run_indices_pgbar,
    summarize_data,
)


@dataclass
class CScaleConvergenceExperimentConfig:
    mode: Literal["run", "postprocess"]
    num_runs: int
    dimensions: int
    popsize: int
    lb: float
    ub: float

    @property
    def bounds(self):
        return (self.lb, self.ub)

    @property
    def result_dir(self):
        return (
            Path(__file__).parent
            / "results"
            / f"d{self.dimensions}_popsize_{self.popsize}_bounds_{int(self.ub)}"
        )

    @classmethod
    def from_omegaconf(cls, cfg: OmegaConf):
        num_runs, dimensions, lb, ub = (
            cfg["num_runs"],  # pyright: ignore[reportIndexIssue]
            cfg["dimensions"],  # pyright: ignore[reportIndexIssue]
            cfg["lb"],  # pyright: ignore[reportIndexIssue]
            cfg["ub"],  # pyright: ignore[reportIndexIssue]
        )  # pyright: ignore[reportIndexIssue]
        match ps := cfg["popsize"]:  # pyright: ignore[reportIndexIssue]
            case "hansen":
                popsize = hansen_cmaes_popsize(dimensions)
            case _:
                raise NotImplementedError(f"Popsize {ps} not supported")
        return cls(
            mode=cfg["mode"],  # pyright: ignore[reportIndexIssue]
            num_runs=num_runs,
            dimensions=dimensions,
            popsize=popsize,
            lb=lb,
            ub=ub,
        )


@dataclass
class CScaleConvergenceExperiment:
    cfg: CScaleConvergenceExperimentConfig

    def run_subprocess(self, run_id: int):
        rng = IndividualGenerator(run_id, self.cfg.bounds, self.cfg.dimensions)
        collector = MetricsCollector(
            [m.CovarianceMatrixNorm(), m.SigmaMeasurement(), m.BestSoFar()], run_id
        )
        fn = get_function_by_name("Elliptic")
        counter = EvalCounter(
            fn,  # pyright: ignore[reportArgumentType]
            bounds=self.cfg.bounds,
            bound_enforcement_method=BoundEnforcement.IGNORE_SOLUTIONS,
        )
        cma = CMAES(
            fun=counter,
            mean=rng.get_individual(),
            popsize=self.cfg.popsize,
            seed=rng.seed,
            stopper=CMAESEarlyStopping(evaluation_budget(self.cfg.dimensions), 1e-11),
            bounds=self.cfg.bounds,
            callbacks=[collector],
        )
        cma.optimize()
        return collector.as_dataframe()

    def run(self):
        dfs = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.run_subprocess)(run_id)
            for run_id in run_indices_pgbar(self.cfg.num_runs)
        )
        raw = pd.concat(dfs)  # pyright: ignore[reportCallIssue, reportArgumentType]
        compress_and_save(raw, self.cfg.result_dir / "raw.parquet")
        summarize_data(raw)
        self.visualize(raw)

    def visualize(self, raw: pd.DataFrame):
        df = raw.copy()
        df["cov_mat_sigma_sq"] = df["cov_mat_norm"] * df["sigma"] ** 2
        actual_inv_hess_norm = np.linalg.norm(
            elliptic_hess_inv_for_dim(self.cfg.dimensions)
        )
        df["cov_mat_ratio"] = df["cov_mat_norm"] / actual_inv_hess_norm
        df["cov_mat_sigma_sq_ratio"] = df["cov_mat_sigma_sq"] / actual_inv_hess_norm

        df = df.reset_index()

        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=False)

        sns.lineplot(
            data=df,
            x="num_evaluations",
            y="cov_mat_ratio",
            estimator="mean",
            errorbar=("pi", 50),
            ax=axes[0],
            label=tex("\\frac{\\|C\\|}{\\|H^{-1}\\|}"),
        )
        sns.lineplot(
            data=df,
            x="num_evaluations",
            y="cov_mat_sigma_sq_ratio",
            estimator="mean",
            errorbar=("pi", 50),
            ax=axes[0],
            label=tex("\\frac{\\|C\\| \\sigma^2}{\\|H^{-1}\\|}"),
        )
        axes[0].axhline(1, linestyle="--", linewidth=1)
        axes[0].set_title(
            "stosunek (przeskalowanej) normy macierzy kowariancji do normy odwrotności hesjanu"
        )
        axes[0].set_xlabel("num_evaluations")
        axes[0].set_ylabel("value")
        axes[0].legend()
        axes[0].set_yscale("log")

        df_long = df.melt(
            id_vars=["num_evaluations", "run_id"],
            value_vars=["sigma", "cov_mat_norm"],
            var_name="metric",
            value_name="value",
        )

        sns.lineplot(
            data=df_long,
            x="num_evaluations",
            y="value",
            hue="metric",
            estimator="mean",
            errorbar=("pi", 50),
            ax=axes[1],
        )
        axes[1].set_title("sigma i norma macierzy kowariancji")
        axes[1].set_xlabel("num_evaluations")
        axes[1].set_ylabel("value")
        axes[1].set_yscale("log")

        sns.lineplot(
            data=df,
            x="num_evaluations",
            y="best",
            estimator="mean",
            errorbar=("pi", 50),
            ax=axes[2],
        )
        axes[2].set_title("krzywa zbieżności")
        axes[2].set_yscale("log")

        for ax in axes:
            ax.grid()

        plt.tight_layout()
        plt.suptitle(f"D={self.cfg.dimensions}, popsize={self.cfg.popsize}")
        plt.savefig(self.cfg.result_dir / "visualization.png", dpi=300)
        plt.show()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: OmegaConf):
    config = CScaleConvergenceExperimentConfig.from_omegaconf(cfg)
    config.result_dir.mkdir(parents=True, exist_ok=True)
    if config.mode == "run":
        exp = CScaleConvergenceExperiment(config)
        exp.run()
    else:
        df = pd.read_parquet(config.result_dir / "raw.parquet")
        exp = CScaleConvergenceExperiment(config)
        exp.visualize(df)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    main()
