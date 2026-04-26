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


@dataclass
class PowellInitialVectorSetPlotter:
    cfg: PowellInitialVectorSetConfig
    save_to_disk: bool = True
    show_special_matrix: bool = False

    def __post_init__(self):
        configure_mpl_for_manuscript()
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        self._raw = pd.read_parquet(self.input_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def input_path(self) -> Path:
        return (
            Path(__file__).parent
            / "results"
            / f"d{self.cfg.dimensions}"
            / "raw.parquet"
        )

    @property
    def output_dir(self) -> Path:
        return Path(__file__).parent / "results" / f"d{self.cfg.dimensions}" / "plots"

    def _save_or_show(self, filename: str):
        if self.save_to_disk:
            plt.savefig(
                self.output_dir / f"{filename}.png", dpi=300, bbox_inches="tight"
            )
        else:
            plt.show()
        plt.close()

    def plot_unaggregated(self):
        _, ax = plt.subplots(figsize=(14, 8))
        colors = plt.cm.tab10.colors

        special_color = "black"
        for run_id, group in self._raw.groupby("run_id"):
            if run_id == -1:
                if not self.show_special_matrix:
                    continue
                color = special_color
            else:
                color = colors[(run_id - 1) % len(colors)]
            default_series = group["best_default"].dropna()
            rotated_series = group["best_rotated_direc"].dropna()
            ax.plot(
                default_series.index,
                default_series.values,
                linestyle="--",
                color=color,
                alpha=0.8,
            )
            ax.plot(
                rotated_series.index,
                rotated_series.values,
                linestyle="-",
                color=color,
                alpha=0.8,
            )
            if run_id != -1:
                no_rotation_series = group["best_no_rotation"].dropna()
                ax.plot(
                    no_rotation_series.index,
                    no_rotation_series.values,
                    linestyle=":",
                    color=color,
                    alpha=0.8,
                )

        ax.plot(
            [], [], linestyle="--", color="gray", label="rotated fn, cartesian basis"
        )
        ax.plot(
            [],
            [],
            linestyle="-",
            color="gray",
            label="rotated fn, rotation eigenvectors",
        )
        ax.plot([], [], linestyle=":", color="gray", label="no rotation")
        if self.show_special_matrix:
            ax.plot(
                [],
                [],
                linestyle="--",
                color=special_color,
                label="special matrix, cartesian basis",
            )
            ax.plot(
                [],
                [],
                linestyle="-",
                color=special_color,
                label="special matrix, rotation eigenvectors",
            )
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
        fig, ax = plt.subplots(figsize=(14, 8))

        groups = dict(list(self._raw.groupby("run_id")))
        regular_groups = {rid: g for rid, g in groups.items() if rid != -1}

        default_series = [g["best_default"].dropna() for g in regular_groups.values()]
        rotated_series = [
            g["best_rotated_direc"].dropna() for g in regular_groups.values()
        ]
        no_rotation_series = [
            g["best_no_rotation"].dropna() for g in regular_groups.values()
        ]

        for series_list, label, color in [
            (default_series, "rotated fn, cartesian basis", "tab:blue"),
            (rotated_series, "rotated fn, rotation eigenvectors", "tab:orange"),
            (no_rotation_series, "no rotation", "tab:green"),
        ]:
            agg = aggregate_convergence_series(series_list)
            ax.plot(agg.index, agg["mean"], label=label, color=color)
            ax.fill_between(agg.index, agg["q25"], agg["q75"], alpha=0.2, color=color)

        if self.show_special_matrix and -1 in groups:
            special = groups[-1]
            special_default = special["best_default"].dropna()
            special_rotated = special["best_rotated_direc"].dropna()
            ax.plot(
                special_default.index,
                special_default.values,
                linestyle="--",
                color="black",
                label="special matrix, cartesian basis",
            )
            ax.plot(
                special_rotated.index,
                special_rotated.values,
                linestyle="-",
                color="black",
                label="special matrix, rotation eigenvectors",
            )

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
    )
    PowellInitialVectorSetPlotter(config).plot_all()


if __name__ == "__main__":
    main()
