from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.find_switch_interval.common import (
    ObjectiveChoice,
    OptimumPosition,
)
from lib.serde import aggregate_dataframes

BASE_DATA_PATH = Path(__file__).parent / "results"


@dataclass
class ConvergencePlotter:
    dimensions: int
    optimum_position: OptimumPosition
    objective_choice: ObjectiveChoice

    def construct_data_path(self) -> Path:
        return (
            BASE_DATA_PATH
            / self.objective_choice.value
            / str(self.dimensions)
            / self.optimum_position.value
            / "raw.parquet"
        )

    def read_data(self):
        return pd.read_parquet(self.construct_data_path())

    def construct_plot_title(self) -> str:
        return f"Krzywa zbieżności CMA-ES ({self.objective_choice.value}, {self.dimensions}D, położenie optimum {self.optimum_position.to_plot_label()})"

    def plot_agg(self):
        df = self.read_data()
        subdf = df[["best", "run_id"]]
        separate = [v for _, v in subdf.groupby("run_id")]
        agg = aggregate_dataframes(separate)  # pyright: ignore[reportArgumentType]
        fig, ax = plt.subplots(figsize=(16, 9))
        agg.plot(ax=ax, y="best", title=self.construct_plot_title(), logy=True)
        plt.legend(["najniższa wartość f. celu"])
        plt.grid()
        plt.show()


if __name__ == "__main__":
    p = ConvergencePlotter(
        50, OptimumPosition.OUTSIDE_CORNER, ObjectiveChoice.RASTRIGIN
    )
    p.plot_agg()
