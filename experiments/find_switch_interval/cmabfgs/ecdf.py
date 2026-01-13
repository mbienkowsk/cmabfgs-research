import os
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cecxx.core import dataclass
from joblib import Parallel, delayed
from scipy.integrate import trapezoid

from experiments.find_switch_interval.cmabfgs.experiment_config import (
    CMABFGSExperimentConfig,
)
from experiments.find_switch_interval.common import ObjectiveChoice, OptimumPosition
from lib.enums import HessianNormalization
from lib.plotting_util import configure_mpl_for_manuscript
from lib.util import compress_and_save, evaluation_budget

ANY_INT = 0


class ECDFMethod(Enum):
    BEST_ACROSS_RUNS = "best_across_runs"
    GLOBAL_OPTIMUM = "global_optimum"


def load_ecdf_from_file(config: CMABFGSExperimentConfig):
    return pd.read_parquet(config.output_directory / "ecdf_curves.parquet")


def agg_ecdf_dir(dim: int, hess_normalization: HessianNormalization):
    return (
        Path(__file__).parent
        / "results"
        / "ecdf_plots"
        / "agg"
        / f"d{dim}"
        / f"{hess_normalization.value}"
    )


def plot_save_path(config: CMABFGSExperimentConfig):
    return (
        Path(__file__).parent
        / "results"
        / "ecdf_plots"
        / config.objective_choice.value
        / str(config.dimensions)
        / config.optimum_position.value
        / f"{config.hess_normalization.value}.png"
    )


def calculate_auc(curve: pd.DataFrame) -> float:
    curve = curve.sort_values("x")
    x = curve["x"].to_numpy()
    y = curve["ecdf"].to_numpy()

    return trapezoid(y, x) / x[-1]


@dataclass
class ECDFCalculator:
    config: CMABFGSExperimentConfig
    n_thresholds: int = 25

    def get_f_ref(self, df: pd.DataFrame, method: ECDFMethod):
        # the postprocessed data already uses the global optimum as reference
        return df["best_so_far"].min() if method == ECDFMethod.BEST_ACROSS_RUNS else 0

    def add_gap_column(self, df: pd.DataFrame, method: ECDFMethod):
        f_ref = self.get_f_ref(df, method)

        df = df.copy()
        df["gap"] = df["best_so_far"] - f_ref
        df["ecdf_method"] = method.value
        return df

    def load_convergence_curves(self) -> pd.DataFrame:
        cmaes_df = pd.read_parquet(self.config.input_file)
        bfgs_df = pd.read_parquet(self.config.output_directory / "raw_curves.parquet")

        df_cmaes_norm = (
            cmaes_df.reset_index()  # num_evaluations is index
            .rename(columns={"best": "best_so_far"})
            .assign(optimizer="CMA-ES")[
                ["num_evaluations", "best_so_far", "run_id", "optimizer"]
            ]
        )

        df_cmabfgs_norm = (
            bfgs_df.query("multiplier != 0")
            .assign(  # drop pure BFGS
                optimizer=lambda d: "CMABFGS-" + d["multiplier"].astype(str)
            )
            .rename(columns={"value": "best_so_far"})[
                ["num_evaluations", "best_so_far", "run_id", "optimizer"]
            ]
        )

        df_all = pd.concat([df_cmaes_norm, df_cmabfgs_norm], ignore_index=True)

        assert (
            df_all.groupby(["optimizer", "run_id"])["best_so_far"]  # pyright: ignore[reportGeneralTypeIssues]
            .apply(lambda s: s.is_monotonic_decreasing)
            .all()
        )

        return df_all  # pyright: ignore[reportReturnType]

    def get_budget_grid(self):
        return np.logspace(
            0,
            np.log10(evaluation_budget(self.config.dimensions)),
            self.n_thresholds,
            endpoint=True,
        )

    @staticmethod
    def compute_ecdf(
        df: pd.DataFrame,
        epsilon: float,
        x_grid: np.ndarray,
        ecdf_method: ECDFMethod,
    ):
        # per run: first hit time
        hit_times = (
            df[df["gap"] <= epsilon]
            .groupby(["optimizer", "run_id"])["num_evaluations"]
            .min()
            .reset_index(name="hit_time")  # pyright: ignore[reportCallIssue]
        )

        # right-censor runs that never hit
        all_runs = df[["optimizer", "run_id"]].drop_duplicates().assign(hit_time=np.inf)

        hit_times = all_runs.merge(
            hit_times, on=["optimizer", "run_id"], how="left"
        ).assign(
            hit_time=lambda d: np.where(
                d["hit_time_y"].notna(),
                d["hit_time_y"],
                d["hit_time_x"],
            )
        )

        # ECDF
        rows = []
        for x in x_grid:
            frac = (hit_times["hit_time"] <= x).groupby(hit_times["optimizer"]).mean()

            rows.append(frac.rename("ecdf").reset_index().assign(x=x, epsilon=epsilon))

        return pd.concat(rows, ignore_index=True)

    def get_eps_grid(self, df: pd.DataFrame, f_ref: float) -> np.ndarray:
        """
        Construct a logarithmic epsilon grid based on empirically achievable gaps.
        """
        # per-run best values
        best_per_run = df.groupby(["optimizer", "run_id"])["best_so_far"].min()

        gaps = best_per_run - f_ref  # pyright: ignore[reportOperatorIssue]

        # remove zero / negative gaps (numerical safety)
        gaps = gaps[gaps > 0]

        if len(gaps) == 0:
            raise ValueError("No positive gaps found; ECDF targets undefined.")

        eps_min = np.percentile(gaps, 10)
        eps_max = np.percentile(gaps, 100)

        # safety guards
        eps_min = max(eps_min, 1e-12)
        # eps_max = max(eps_max, eps_min * 10)

        return np.logspace(
            np.log10(eps_min),
            np.log10(eps_max),
            self.n_thresholds,
        )

    def compute_all_ecdfs(self):
        df_base = self.load_convergence_curves()
        x_grid = self.get_budget_grid()

        ecdfs = []

        for method in ECDFMethod:
            df_m = self.add_gap_column(df_base, method)
            f_ref = self.get_f_ref(df_base, method)
            grid = self.get_eps_grid(df_m, f_ref)  # pyright: ignore[reportArgumentType]

            for eps in grid:
                ecdf = self.compute_ecdf(df_m, eps, x_grid, method)
                ecdf["ecdf_method"] = method.value
                ecdfs.append(ecdf)

        ecdf_df = pd.concat(ecdfs, ignore_index=True)

        return ecdf_df

    def run(self):
        ecdf_df = self.compute_all_ecdfs()
        compress_and_save(ecdf_df, self.config.output_directory / "ecdf_curves.parquet")


def process_config(config: CMABFGSExperimentConfig):
    ECDFCalculator(config).run()
    ecdf = load_ecdf_from_file(config)
    plot_ecdf(
        ecdf,
        f"d={config.dimensions}, F={config.objective_choice.value}",
        ECDFMethod.GLOBAL_OPTIMUM,
        plot_save_path(config),
        False,
    )


def plot_ecdf(
    ecdf_df: pd.DataFrame,
    title: str,
    method: ECDFMethod,
    save_path: Path,
    show: bool = False,
    label_fn: Callable[[str], str] | None = None,
):
    configure_mpl_for_manuscript()

    frame = (
        ecdf_df.query("ecdf_method == @method.value")
        .groupby(["optimizer", "x"])["ecdf"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots()

    for optimizer, g in frame.groupby("optimizer"):
        label = label_fn(optimizer) if label_fn else optimizer  # pyright: ignore[reportArgumentType]
        ax.plot(g["x"], g["ecdf"], label=label)

    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Fraction of runs solved")
    ax.grid()
    ax.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_cec_ecdfs_with_auc(
    dimensions: int,
    hess_normalization: HessianNormalization,
    method: ECDFMethod,
    save_dir: Path,
):
    ecdf_frames = []
    save_dir.mkdir(parents=True, exist_ok=True)

    for obj in ObjectiveChoice.all_cec_objectives():
        config = CMABFGSExperimentConfig(
            dimensions,
            ANY_INT,
            obj,
            OptimumPosition.MIDDLE,
            False,
            hess_normalization,
        )

        ecdf_path = config.output_directory / "ecdf_curves.parquet"
        if not ecdf_path.exists():
            continue

        ecdf = pd.read_parquet(ecdf_path)
        ecdf_frames.append(ecdf)

    if not ecdf_frames:
        raise RuntimeError("No ECDF data found to plot")

    # aggregate ECDFs across CEC functions
    ecdf_all = (
        pd.concat(ecdf_frames, ignore_index=True)
        .groupby(["optimizer", "x", "epsilon", "ecdf_method"])["ecdf"]
        .mean()
        .reset_index()
    )
    compress_and_save(ecdf_all, save_dir / "aggregated_ecdf.parquet")

    # compute AUC per optimizer (averaged over epsilons)
    auc_map = (
        ecdf_all.query("ecdf_method == @method.value")
        .groupby(["optimizer", "epsilon"])
        .apply(calculate_auc)
        .reset_index(name="auc")  # pyright: ignore[reportCallIssue]
        .groupby("optimizer")["auc"]
        .mean()
        .to_dict()
    )

    def label_fn(optimizer: str) -> str:
        return f"{optimizer} (AUC={auc_map[optimizer]:.3f})"

    plot_ecdf(
        ecdf_df=ecdf_all,
        title=f"ECDF (d={dimensions}, hess={hess_normalization.value}, {method.value})",
        method=method,
        save_path=save_dir / "agg_ecdf.png",
        show=False,
        label_fn=label_fn,
    )


if __name__ == "__main__":
    debug = bool(os.getenv("DEBUG", ""))
    print(f"Debug mode: {debug}")

    if debug:
        config = CMABFGSExperimentConfig(
            100,
            ANY_INT,
            ObjectiveChoice.CEC17,
            OptimumPosition.MIDDLE,
            True,
            HessianNormalization.UNIT,
        )
        ecdf = ECDFCalculator(config).compute_all_ecdfs()
        method = ECDFMethod.GLOBAL_OPTIMUM
        plot_ecdf(
            ecdf,
            f"d={config.dimensions}, F={config.objective_choice.value}",
            method,
            plot_save_path(config),
            True,
        )

    else:
        hess_norms = [
            # HessianNormalization.UNIT_DIM,
            HessianNormalization.UNIT,
        ]

        # CEC configurations
        cec_optimum_positions = [
            OptimumPosition.MIDDLE,
        ]
        # cec_dims = [10, 30, 50, 100]
        cec_dims = [100]
        cec_objectives = ObjectiveChoice.all_cec_objectives()
        cec_configurations = [
            CMABFGSExperimentConfig(
                d, ANY_INT, obj, OptimumPosition.MIDDLE, False, hess_norm
            )
            for d, obj, opt, hess_norm in product(
                cec_dims, cec_objectives, cec_optimum_positions, hess_norms
            )
        ]

        Parallel(n_jobs=-1, backend="loky")(
            delayed(process_config)(config) for config in cec_configurations
        )
        Parallel(n_jobs=-1, backend="loky")(
            delayed(plot_cec_ecdfs_with_auc)(
                d,
                hess_norm,
                ECDFMethod.GLOBAL_OPTIMUM,
                agg_ecdf_dir(d, hess_norm),
            )
            for d, hess_norm in product(cec_dims, hess_norms)
        )
