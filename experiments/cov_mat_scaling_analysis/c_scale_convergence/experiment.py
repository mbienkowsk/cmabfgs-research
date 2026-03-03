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
    bounds: tuple[float, float]

    @property
    def result_dir(self):
        return (
            Path(__file__).parent
            / "results"
            / f"d{self.dimensions}_popsize_{self.popsize}_bounds_{int(self.bounds[1])}"
        )

    @classmethod
    def from_omegaconf(cls, cfg: OmegaConf):
        num_runs, dimensions, bounds = (
            cfg["num_runs"],  # pyright: ignore[reportIndexIssue]
            cfg["dimensions"],  # pyright: ignore[reportIndexIssue]
            cfg["bounds"],  # pyright: ignore[reportIndexIssue]
        )  # pyright: ignore[reportIndexIssue]
        match ps := cfg["popsize"]:  # pyright: ignore[reportIndexIssue]
            case "hansen":
                popsize = hansen_cmaes_popsize(dimensions)
            case "beyer":
                popsize = 4 * dimensions
            case _:
                raise NotImplementedError(f"Popsize {ps} not supported")
        return cls(
            mode=cfg["mode"],  # pyright: ignore[reportIndexIssue]
            num_runs=num_runs,
            dimensions=dimensions,
            popsize=popsize,
            bounds=(-bounds, bounds),
        )


@dataclass
class CScaleConvergenceExperiment:
    cfg: CScaleConvergenceExperimentConfig

    def run_subprocess(self, run_id: int):
        rng = IndividualGenerator(run_id, self.cfg.bounds, self.cfg.dimensions)
        collector = MetricsCollector(
            [
                m.CovarianceMatrixNorm(),
                m.SigmaMeasurement(),
                m.BestSoFar(),
                m.CovarianceMatrixEigenvalueList(),
            ],
            run_id,
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

        fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=False)
        ax_0 = axes[0][0]

        sns.lineplot(
            data=df,
            x="num_evaluations",
            y="cov_mat_ratio",
            estimator="mean",
            errorbar=("pi", 50),
            ax=ax_0,
            label=tex("\\frac{\\|C\\|}{\\|H^{-1}\\|}"),
        )
        sns.lineplot(
            data=df,
            x="num_evaluations",
            y="cov_mat_sigma_sq_ratio",
            estimator="mean",
            errorbar=("pi", 50),
            ax=ax_0,
            label=tex("\\frac{\\|C\\| \\sigma^2}{\\|H^{-1}\\|}"),
        )
        ax_0.axhline(1, linestyle="--", linewidth=1)
        ax_0.set_title(
            "stosunek (przeskalowanej) normy macierzy kowariancji do normy odwrotności hesjanu"
        )
        ax_0.set_xlabel("num_evaluations")
        ax_0.set_ylabel("value")
        ax_0.legend()
        ax_0.set_yscale("log")

        df_long = df.melt(
            id_vars=["num_evaluations", "run_id"],
            value_vars=["sigma", "cov_mat_norm"],
            var_name="metric",
            value_name="value",
        )
        ax_0.grid()

        ax_1 = axes[0][1]
        sns.lineplot(
            data=df_long,
            x="num_evaluations",
            y="value",
            hue="metric",
            estimator="mean",
            errorbar=("pi", 50),
            ax=ax_1,
        )
        ax_1.set_title("sigma i norma macierzy kowariancji")
        ax_1.set_xlabel("num_evaluations")
        ax_1.set_ylabel("value")
        ax_1.set_yscale("log")
        ax_1.grid()

        ax_2 = axes[1][0]
        sns.lineplot(
            data=df,
            x="num_evaluations",
            y="best",
            estimator="mean",
            errorbar=("pi", 50),
            ax=ax_2,
        )
        ax_2.set_title("krzywa zbieżności")
        ax_2.set_yscale("log")
        ax_2.grid()

        ax_3 = axes[1][1]
        eigvals = np.stack(df["cov_mat_eigv"].to_numpy())  # pyright: ignore[reportCallIssue, reportArgumentType]
        eigvals_df = pd.DataFrame(
            eigvals,
            columns=[f"eig_{i}" for i in range(eigvals.shape[1])],  # pyright: ignore[reportArgumentType]
        )
        df_expanded = pd.concat(
            [df.drop(columns=["cov_mat_eigv"]).reset_index(drop=True), eigvals_df],
            axis=1,
        )
        df_long = df_expanded.melt(
            id_vars=["num_evaluations", "run_id"],
            value_vars=[c for c in df_expanded.columns if c.startswith("eig_")],
            var_name="component",
            value_name="eigenvalue",
        )
        if eigvals.shape[1] <= 10:
            sns.lineplot(
                data=df_long,
                x="num_evaluations",
                y="eigenvalue",
                hue="component",
                estimator="mean",
                errorbar=("pi", 50),
                ax=ax_3,
            )
            ax_3.set_title("wartości własne")
        else:
            sns.lineplot(
                data=df_long,
                x="num_evaluations",
                y="eigenvalue",
                estimator="mean",
                errorbar=("pi", 50),
                ax=ax_3,
            )
            ax_3.set_title("zagregowane wartości własne")

        ax_3.grid()
        ax_3.set_yscale("log")

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
