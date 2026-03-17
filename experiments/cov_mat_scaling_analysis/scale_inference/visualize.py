from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra
import matplotlib.pyplot as plt
import pandas as pd

from config.schema import MasterConfig
from lib.plotting_util import (
    configure_mpl_for_manuscript,
    plot_with_legend_function,
    set_log_x_labels,
    tex,
)
from lib.serde import aggregate_dataframes


@dataclass
class CScaleInferenceConfig:
    dimensions: int
    objective: str
    bounds: int


_COL_RE = re.compile(r"^best_(\d+)_(identity|C)_(unit|rescaled)$")

MatrixType = Literal["identity", "C"]
ScalingType = Literal["unit", "rescaled"]


def parse_column(col: str) -> tuple[int, MatrixType, ScalingType] | None:
    m = _COL_RE.match(col)
    if not m:
        return None
    return int(m.group(1)), m.group(2), m.group(3)  # ty: ignore[invalid-return-type]


def extract_iteration_numbers(df: pd.DataFrame) -> list[int]:
    iters = set()
    for col in df.columns:
        parsed = parse_column(col)
        if parsed is not None:
            iters.add(parsed[0])
    return sorted(iters)


def select_columns_for_iter(df: pd.DataFrame, iteration: int) -> pd.DataFrame:
    cols = [
        col
        for col in df.columns
        if parse_column(col) is not None and parse_column(col)[0] == iteration  # ty: ignore[not-subscriptable]
    ]
    return df[cols]


_LABEL_MAP: dict[tuple[MatrixType, ScalingType], str] = {
    ("identity", "unit"): tex("I") + ", unit",
    ("identity", "rescaled"): tex("I") + ", rescaled",
    ("C", "unit"): tex("C") + ", unit",
    ("C", "rescaled"): tex("C") + ", rescaled",
}


def column_label(col: str) -> str:
    parsed = parse_column(col)
    if parsed is None:
        return col
    _, matrix, scaling = parsed
    return _LABEL_MAP.get((matrix, scaling), col)


@dataclass
class CScaleInferencePlotter:
    cfg: CScaleInferenceConfig
    save_to_disk: bool = True

    def __post_init__(self):
        configure_mpl_for_manuscript()

        if not self.input_file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file_path}")

        self._raw = pd.read_parquet(self.input_file_path)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    @property
    def input_file_path(self) -> Path:
        return (
            Path(__file__).parent
            / "results"
            / f"d{self.cfg.dimensions}"
            / f"{self.cfg.objective}_{self.cfg.bounds}"
            / "raw.parquet"
        )

    @property
    def output_directory(self) -> Path:
        return (
            Path(__file__).parent
            / "results"
            / f"d{self.cfg.dimensions}"
            / f"{self.cfg.objective}_{self.cfg.bounds}"
            / "plots"
        )

    def _build_aggregated(self) -> pd.DataFrame:
        """
        Split raw data by run_id, keep only the value columns (drop run_id),
        set the row index to the DataFrame's integer index (which represents
        evaluation count), then interpolate onto a common grid and mean.
        """
        run_dfs = []
        for _, group in self._raw.groupby("run_id"):
            value_cols = [c for c in group.columns if parse_column(c) is not None]
            run_df = group[value_cols].copy()
            run_dfs.append(run_df)

        return aggregate_dataframes(run_dfs, drop_col=None)

    @contextmanager
    def _new_ax(self, for_iteration: int):
        fig, ax = plt.subplots(figsize=(16, 9))
        set_log_x_labels(ax)
        yield ax
        cfg = self.cfg
        ax.set_xlabel("Liczba ewaluacji funkcji celu")
        ax.set_ylabel("f(x_best)")
        ax.set_title(f"$d={cfg.dimensions}$, iter={for_iteration}")
        ax.grid()
        ax.set_yscale("log")
        plt.tight_layout()

    def _finalize(self, filename: str):
        if self.save_to_disk:
            plt.savefig(
                self.output_directory / f"{filename}.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.show()
        plt.close()

    def plot_for_iteration(self, agg: pd.DataFrame, iteration: int):
        """
        One plot per iteration number.  Four lines:
          - identity / unit
          - identity / rescaled
          - C / unit
          - C / rescaled
        """
        data = select_columns_for_iter(agg, iteration)
        if data.empty:
            return

        with self._new_ax(iteration) as ax:
            plot_with_legend_function(data, ax, column_label)

        self._finalize(f"iter_{iteration}")

    def plot_all(self):
        agg = self._build_aggregated()
        for iteration in extract_iteration_numbers(self._raw):
            self.plot_for_iteration(agg, iteration)


@hydra.main(version_base=None, config_name="config", config_path="../../../config/")
def main(cfg: MasterConfig) -> None:
    plotter = CScaleInferencePlotter(
        cfg=cfg.experiments.c_scale_inference, save_to_disk=True
    )
    plotter.plot_all()


if __name__ == "__main__":
    main()
