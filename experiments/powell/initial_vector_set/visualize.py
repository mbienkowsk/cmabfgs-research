from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd

from config.paths import CONFIG_DIR_STR
from config.schema import MasterConfig
from experiments.powell.initial_vector_set.experiment import (
    PowellInitialVectorSetConfig,
)
from lib.plotting_util import configure_mpl_for_manuscript
from lib.serde import aggregate_convergence_series

_METHODS = [
    ("best_default", "rotated fn, cartesian basis", "--", "tab:blue"),
    ("best_rotated_direc", "rotated fn, rotation eigenvectors", "-", "tab:orange"),
    ("best_no_rotation", "no rotation", ":", "tab:green"),
]


@dataclass
class PowellInitialVectorSetPlotter:
    cfg: PowellInitialVectorSetConfig
    save_to_disk: bool = True

    def __post_init__(self):
        configure_mpl_for_manuscript()
        if not self.processed_path.exists():
            raise FileNotFoundError(f"Processed data not found: {self.processed_path}")
        self._processed = pd.read_parquet(self.processed_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def processed_path(self) -> Path:
        return Path(__file__).parent / "results" / f"d{self.cfg.dimensions}" / "processed.parquet"

    @property
    def output_dir(self) -> Path:
        return Path(__file__).parent / "results" / f"d{self.cfg.dimensions}" / "plots"

    def _save_or_show(self, filename: str):
        if self.save_to_disk:
            plt.savefig(self.output_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_unaggregated(self):
        _, ax = plt.subplots(figsize=(14, 8))
        colors = plt.cm.tab10.colors

        for matrix_id, matrix_df in self._processed.groupby("matrix_id"):
            color = colors[(matrix_id - 1) % len(colors)]
            for method, _, linestyle, _ in _METHODS:
                series = matrix_df[method].dropna()
                ax.plot(series.index, series.values, linestyle=linestyle, color=color, alpha=0.8)

        for _, label, linestyle, _ in _METHODS:
            ax.plot([], [], linestyle=linestyle, color="gray", label=label)

        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("f(x_best)")
        ax.set_title(f"Powell — per-matrix convergence, $d={self.cfg.dimensions}$")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        self._save_or_show("unaggregated")

    def plot_aggregated(self):
        _, ax = plt.subplots(figsize=(14, 8))

        for method, label, linestyle, color in _METHODS:
            series_list = [
                matrix_df[method].dropna()
                for _, matrix_df in self._processed.groupby("matrix_id")
            ]
            agg = aggregate_convergence_series(series_list)
            ax.plot(agg.index, agg["mean"], linestyle=linestyle, color=color, label=label)
            ax.fill_between(agg.index, agg["q25"], agg["q75"], alpha=0.2, color=color)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("f(x_best)")
        ax.set_title(f"Powell — aggregated convergence, $d={self.cfg.dimensions}$")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        self._save_or_show("aggregated")

    def plot_all(self):
        self.plot_unaggregated()
        self.plot_aggregated()


@hydra.main(version_base=None, config_name="config", config_path=CONFIG_DIR_STR)
def main(cfg: MasterConfig) -> None:
    exp_cfg = cfg.experiments.powell_initial_vector_set
    config = PowellInitialVectorSetConfig(
        dimensions=exp_cfg.dimensions,
        bounds=exp_cfg.bounds,
        num_matrices=exp_cfg.num_matrices,
        num_starting_points=exp_cfg.num_starting_points,
    )
    PowellInitialVectorSetPlotter(config).plot_all()


if __name__ == "__main__":
    main()
