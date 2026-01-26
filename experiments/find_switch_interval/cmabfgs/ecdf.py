import os
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

THRESHOLD_MIN = 1e-8


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
        / f"{config.hess_normalization.value}.svg"
    )


def calculate_auc(curve: pd.DataFrame) -> float:
    curve = curve.sort_values("x")
    x = curve["x"].to_numpy()
    y = curve["ecdf"].to_numpy()

    return trapezoid(y, x) / x[-1]


@dataclass
class ECDFCalculator:
    config: CMABFGSExperimentConfig
    n_thresholds: int = 50

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
    def _compute_ecdf(
        df: pd.DataFrame,
        thresholds: np.ndarray,
        x_grid: np.ndarray,
    ) -> pd.DataFrame:
        rows = []

        for (optimizer, run_id), g in df.groupby(["optimizer", "run_id"]):  # pyright: ignore[reportGeneralTypeIssues]
            g = g.sort_values("num_evaluations")

            fracs = g["best_so_far"].apply(
                lambda gap: np.sum(gap <= thresholds) / len(thresholds)
            )

            run_curve = pd.DataFrame(
                {
                    "optimizer": optimizer,
                    "run_id": run_id,
                    "num_evaluations": g["num_evaluations"].values,
                    "frac": fracs.values,
                }
            )

            # discretize
            for x in x_grid:
                past = run_curve[run_curve["num_evaluations"] <= x]
                y = past["frac"].max() if not past.empty else 0.0

                rows.append(
                    {
                        "optimizer": optimizer,
                        "x": x,
                        "ecdf": y,
                    }
                )

        return (
            pd.DataFrame(rows)
            .groupby(["optimizer", "x"], as_index=False)["ecdf"]
            .mean()
        )

    def get_threshold_grid(self, df: pd.DataFrame):
        tmin = THRESHOLD_MIN
        tmax = df["best_so_far"].max()
        return np.logspace(np.log10(tmin), np.log10(tmax), self.n_thresholds)

    def compute_ecdf(self):
        df = self.load_convergence_curves()
        x_grid = self.get_budget_grid()
        threshold_grid = self.get_threshold_grid(df)
        ecdf = self._compute_ecdf(df, threshold_grid, x_grid)

        return ecdf

    def run(self):
        ecdf_df = self.compute_ecdf()
        compress_and_save(ecdf_df, self.config.output_directory / "ecdf_curves.parquet")


def process_config(config: CMABFGSExperimentConfig):
    ECDFCalculator(config).run()
    ecdf = load_ecdf_from_file(config)
    plot_ecdf(
        ecdf,
        f"d={config.dimensions}, F={config.objective_choice.value}",
        plot_save_path(config),
        False,
    )


def plot_ecdf(
    ecdf_df: pd.DataFrame,
    title: str,
    save_path: Path,
    show: bool = False,
    label_fn: Callable[[str], str] | None = None,
):
    configure_mpl_for_manuscript()

    frame = ecdf_df.groupby(["optimizer", "x"])["ecdf"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(16, 9))

    for optimizer, g in frame.groupby("optimizer"):
        label = label_fn(optimizer) if label_fn else optimizer  # pyright: ignore[reportArgumentType]
        ax.plot(g["x"], g["ecdf"], label=label)

    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Liczba ewaluacji funkcji celu")
    ax.set_ylabel("Ułamek osiągniętych poziomów satysfakcji")
    ax.grid()
    ax.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_cec_ecdfs_with_auc(
    dimensions: int,
    hess_normalization: HessianNormalization,
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
        .groupby(["optimizer", "x"])["ecdf"]
        .mean()
        .reset_index()
    )
    compress_and_save(ecdf_all, save_dir / "aggregated_ecdf.parquet")

    # compute AUC per optimizer
    auc_series = (
        ecdf_all.groupby(["optimizer"])
        .apply(calculate_auc)
        .reset_index(name="auc")  # pyright: ignore[reportCallIssue]
        .groupby("optimizer")["auc"]
        .mean()
    )

    auc_to_disk = (
        auc_series.to_frame(name="auc")
        .reset_index()
        .assign(
            dimension=dimensions,
            hess_normalization=hess_normalization.value,
        )
    )
    auc_to_disk["auc_norm"] = auc_to_disk["auc"] / auc_to_disk.groupby("dimension")[
        "auc"
    ].transform("max")
    auc_to_disk.to_csv(save_dir / "auc.csv", index=False)

    auc_map = auc_series.to_dict()

    def label_fn(optimizer: str) -> str:
        return f"{optimizer} (AUC={auc_map[optimizer]:.3f})"

    plot_ecdf(
        ecdf_df=ecdf_all,
        title=f"ECDF (d={dimensions})",
        save_path=save_dir / "agg_ecdf.svg",
        show=False,
        label_fn=label_fn,
    )


if __name__ == "__main__":
    debug = bool(os.getenv("DEBUG", ""))
    print(f"Debug mode: {debug}")

    if debug:
        # plot_cec_ecdfs_with_auc(
        #     100,
        #     HessianNormalization.UNIT,
        #     agg_ecdf_dir(100, HessianNormalization.UNIT),
        # )
        config = CMABFGSExperimentConfig(
            100,
            ANY_INT,
            ObjectiveChoice.CEC17,
            OptimumPosition.MIDDLE,
            True,
            HessianNormalization.UNIT,
        )
        ecdf = ECDFCalculator(config).compute_ecdf()
        plot_ecdf(
            ecdf,
            f"d={config.dimensions}, F={config.objective_choice.value}",
            plot_save_path(config),
            True,
        )
    else:
        hess_norms = [
            # HessianNormalization.UNIT_DIVIDED_BY_DIM_ROOT,
            HessianNormalization.UNIT,
        ]

        # CEC configurations
        cec_optimum_positions = [
            OptimumPosition.MIDDLE,
        ]
        cec_dims = [10, 30, 50, 100]
        # cec_dims = [100]

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
                agg_ecdf_dir(d, hess_norm),
            )
            for d, hess_norm in product(cec_dims, hess_norms)
        )
