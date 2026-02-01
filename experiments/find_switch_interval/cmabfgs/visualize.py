import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

from experiments.find_switch_interval.cmabfgs.experiment_config import (
    CMABFGSExperimentConfig,
)
from experiments.find_switch_interval.common import ObjectiveChoice, OptimumPosition
from lib.enums import HessianNormalization
from lib.plotting_util import configure_mpl_for_manuscript, set_log_x_labels
from lib.util import evaluation_budget

ANY_INT = 0


@dataclass
class CMABFGSPlotter:
    config: CMABFGSExperimentConfig
    with_removed_outliers: bool = False
    save_to_disk: bool = True

    @property
    def raw_curves_input_file(self):
        filename = (
            "raw_curves.parquet"
            if not self.with_removed_outliers
            else "raw_curves_no_outliers.parquet"
        )
        return self.config.output_directory / filename

    @property
    def agg_curves_input_file(self):
        filename = (
            "agg_curves.parquet"
            if not self.with_removed_outliers
            else "agg_curves_no_outliers.parquet"
        )
        return self.config.output_directory / filename

    @property
    def plot_save_path(self):
        filename = (
            f"{self.config.hess_normalization.value}.png"
            if not self.with_removed_outliers
            else f"{self.config.hess_normalization.value}_no_outliers.png"
        )
        return (
            Path(__file__).parent
            / "results"
            / "plots"
            / self.config.objective_choice.value
            / str(self.config.dimensions)
            / self.config.optimum_position.value
            / filename
        )

    @property
    def cmaes_input_file(self):
        return self.config.input_file.parent / "agg.parquet"

    def load_agg_df(self):
        df = pd.read_parquet(self.agg_curves_input_file)
        return df[df["num_evaluations"] < evaluation_budget(self.config.dimensions)]

    def load_cmaes_df(self):
        return pd.read_parquet(self.cmaes_input_file)

    def get_label_from_mul(self, mul: float) -> str:
        if mul == 0:
            return "BFGS"
        return f"CMABFGS, $k={mul}$"

    def plot(self, agg_df: pd.DataFrame, cmaes_df: pd.DataFrame):
        configure_mpl_for_manuscript()
        fig, ax = plt.subplots(figsize=(16, 11))
        set_log_x_labels(ax)

        secax = ax.secondary_xaxis(
            "bottom",
            functions=(
                lambda x: x / (4 * self.config.dimensions),  # pyright: ignore[reportOperatorIssue]
                lambda x: x * 4 * self.config.dimensions,  # pyright: ignore[reportOperatorIssue]
            ),
        )
        secax.spines["bottom"].set_position(("outward", 80))

        for mul, mul_df in agg_df.groupby("multiplier"):
            mul_df.plot(
                ax=ax,
                logy=True,
                x="num_evaluations",
                y="mean",
                label=self.get_label_from_mul(mul),  # pyright: ignore[reportArgumentType]
            )
            # ax.fill_between(
            #     mul_df["num_evaluations"], mul_df["q25"], mul_df["q75"], alpha=0.4
            # )

        cmaes_df.plot(ax=ax, logy=True, y="best_cmaes", label="CMA-ES")
        ax.grid()
        fun_name = (
            self.config.objective_choice.value
            if self.config.objective_choice != ObjectiveChoice.ELLIPTIC
            else "SDP"
        )
        title = f"d={self.config.dimensions}, f={fun_name}"
        if not self.config.objective_choice.value.startswith("CEC"):
            title += f"\n optimum {self.config.optimum_position.to_plot_label()}"

        plt.title(title)
        secax.set_xlabel("Iteracje CMA-ES")
        ax.set_xlabel("Liczba ewaluacji funkcji celu")
        plt.tight_layout()
        if self.config.debug:
            plt.show()
        else:
            self.plot_save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(self.plot_save_path, dpi=300)
        plt.close()

    def run(self):
        agg_df = self.load_agg_df()
        cmaes_df = self.load_cmaes_df()
        self.plot(agg_df, cmaes_df)


def plot_config(config: CMABFGSExperimentConfig, with_removed_outliers: bool):
    plotter = CMABFGSPlotter(
        config, save_to_disk=True, with_removed_outliers=with_removed_outliers
    )
    plotter.run()


if __name__ == "__main__":
    debug = bool(os.getenv("DEBUG", ""))
    print(f"Debug mode: {debug}")

    if debug:
        config = CMABFGSExperimentConfig(
            100,
            ANY_INT,
            ObjectiveChoice.RASTRIGIN,
            OptimumPosition.MIDDLE,
            True,
            HessianNormalization.UNIT,
        )
        CMABFGSPlotter(config, with_removed_outliers=True, save_to_disk=False).run()

    else:
        REMOVE_OUTLIERS = True
        cec_optimum_positions = [
            OptimumPosition.MIDDLE,
        ]
        hess_norms = [
            HessianNormalization.UNIT,
        ]

        # cec dims vary from control dims
        cec_dims = [10, 30, 50, 100]
        cec_objectives = [getattr(ObjectiveChoice, f"CEC{i}") for i in range(1, 31)]
        cec_configurations = [
            CMABFGSExperimentConfig(d, ANY_INT, obj, opt, False, hess_norm)
            for d, obj, opt, hess_norm in product(
                cec_dims, cec_objectives, cec_optimum_positions, hess_norms
            )
        ]

        control_optimum_positions = list(OptimumPosition)
        control_dims = [10, 20, 50, 100]
        control_objectives = [
            ObjectiveChoice.ELLIPTIC,
            ObjectiveChoice.RASTRIGIN,
        ]
        control_configurations = [
            CMABFGSExperimentConfig(d, ANY_INT, obj, opt, False, hess_norm)
            for d, obj, opt, hess_norm in product(
                control_dims, control_objectives, control_optimum_positions, hess_norms
            )
        ]
        # all_configurations = cec_configurations + control_configurations
        all_configurations = control_configurations

        Parallel(n_jobs=-1, backend="loky")(
            delayed(plot_config)(config, REMOVE_OUTLIERS)
            for config in all_configurations
        )
