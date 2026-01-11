from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.find_switch_interval.cmabfgs.experiment_config import (
    CMABFGSExperimentConfig,
)
from experiments.find_switch_interval.common import ObjectiveChoice, OptimumPosition
from lib.enums import HessianNormalization

ANY_INT = 0


@dataclass
class CMABFGSPlotter:
    config: CMABFGSExperimentConfig
    save_to_disk: bool = True

    @property
    def raw_curves_input_file(self):
        return self.config.output_directory / "raw_curves.parquet"

    @property
    def agg_curves_input_file(self):
        return self.config.output_directory / "agg_curves.parquet"

    @property
    def plot_save_path(self):
        return (
            Path(__file__).parent
            / "results"
            / "plots"
            / self.config.objective_choice.value
            / str(self.config.dimensions)
            / self.config.optimum_position.value
            / f"{self.config.hess_normalization.value}.png"
        )

    @property
    def cmaes_input_file(self):
        return self.config.input_file.parent / "agg.parquet"

    def load_agg_df(self):
        return pd.read_parquet(self.agg_curves_input_file)

    def load_cmaes_df(self):
        return pd.read_parquet(self.cmaes_input_file)

    def get_label_from_mul(self, mul: float) -> str:
        if mul == 0:
            return "Vanilla BFGS"
        iters = int(self.config.dimensions * mul)
        return f"Przełączenie co {iters} iteracji"

    def plot(self, agg_df: pd.DataFrame, cmaes_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(16, 11))

        secax = ax.secondary_xaxis(
            "bottom",
            functions=(
                lambda x: x / (4 * self.config.dimensions),  # pyright: ignore[reportOperatorIssue]
                lambda x: x * 4 * self.config.dimensions,  # pyright: ignore[reportOperatorIssue]
            ),
        )
        secax.spines["bottom"].set_position(("outward", 40))

        for mul, mul_df in agg_df.groupby("multiplier"):
            mul_df.plot(
                ax=ax,
                logy=True,
                x="num_evaluations",
                y="mean",
                label=self.get_label_from_mul(mul),  # pyright: ignore[reportArgumentType]
            )

        cmaes_df.plot(ax=ax, logy=True, y="best_cmaes", label="vanilla CMA-ES")
        ax.grid()

        plt.title(
            f"Krzywe zbieżności CMA-ES i CMABFGS (d={self.config.dimensions}, f={self.config.objective_choice.value}, optimum {self.config.optimum_position.to_plot_label()})\n\
            Hesjan: {self.config.hess_normalization.to_plot_label()}"
        )
        secax.set_xlabel("Iteracje CMA-ES")
        ax.set_xlabel("Liczba ewaluacji funkcji celu")
        plt.tight_layout()
        if self.config.debug:
            plt.show()
        else:
            self.plot_save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(self.plot_save_path, dpi=300)

    def run(self):
        agg_df = self.load_agg_df()
        cmaes_df = self.load_cmaes_df()
        self.plot(agg_df, cmaes_df)


if __name__ == "__main__":
    config = CMABFGSExperimentConfig(
        100,
        ANY_INT,
        ObjectiveChoice.CEC1,
        OptimumPosition.MIDDLE,
        True,
        HessianNormalization.UNIT,
    )
    CMABFGSPlotter(config, save_to_disk=False).run()
