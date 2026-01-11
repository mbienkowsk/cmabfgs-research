import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from lib.enums import HessianNormalization
from lib.plotting_util import (
    configure_mpl_for_manuscript,
    plot_with_legend_function,
    set_log_x_labels,
    tex,
)
from lib.util import trim_constant_tail


def extract_iterations_from_column(col: str):
    m = re.search(r"\d+", col)
    if not m:
        return None
    return int(m.group(0))


def extract_normalization_from_column(col: str):
    for norm in HessianNormalization:
        if col.endswith(norm.value):
            return norm
    return None


@dataclass
class BFGSAccelerationPlotter:
    dimensions: int
    x0_mode: Literal["random", "inherited"]
    manuscript_version: bool = True
    save_to_disk: bool = True

    def __post_init__(self):
        if self.manuscript_version:
            configure_mpl_for_manuscript()

        if not self.input_file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file_path}")

        data = pd.read_parquet(self.input_file_path)

        self.df = pd.concat(
            {c: trim_constant_tail(data[c]) for c in data.columns},  # pyright: ignore[reportArgumentType]
            axis=1,
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)

    @property
    def input_file_path(self):
        return (
            Path(__file__).parent
            / "results"
            / f"d{self.dimensions}"
            / "agg"
            / (
                "bfgs.parquet"
                if self.x0_mode == "random"
                else "bfgs_inherited_x0.parquet"
            )
        )

    @property
    def output_directory(self):
        return (
            Path(__file__).parent
            / "results"
            / "plots"
            / f"d{self.dimensions}"
            / ("random_x0" if self.x0_mode == "random" else "inherited_x0")
        )

    def get_data_for_normalization_method(self, norm: HessianNormalization):
        return cast(
            pd.DataFrame,
            self.df[[col for col in self.df.columns if col.endswith(norm.value)]],
        )

    def get_data_for_hess_iteration(self, iter: int):
        cols = [
            col
            for col in self.df.columns
            if extract_iterations_from_column(col) == iter
        ]
        if not cols:
            all_iters = set(extract_iterations_from_column(col) for col in self.df)
            raise ValueError(
                f"No BFGS preconditioned with the matrix for input {iter}. Possible values: {all_iters}"
            )
        return cast(pd.DataFrame, self.df[cols])

    @contextmanager
    def new_ax(self):
        fig, ax = plt.subplots(figsize=(16, 9))
        set_log_x_labels(ax)

        yield ax

        ax.set_xlabel("Liczba ewaluacji funkcji celu")
        ax.set_ylabel("f(xbest)")
        ax.grid()
        ax.set_yscale("log")
        plt.tight_layout()

    def plot_comparison_for_norm(self, norm: HessianNormalization):
        data = self.get_data_for_normalization_method(norm)

        def to_label(col: str):
            if (iters := extract_iterations_from_column(col)) is not None:
                return str(iters)
            return "I" if "identity" in col else "H_inv" if "inv_hess" in col else col

        with self.new_ax() as ax:
            plot_with_legend_function(data, ax, to_label)
            plt.title(norm.to_plot_label())

        if self.save_to_disk:
            plt.savefig(
                self.output_directory / f"by_iteration_{norm.value}.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def plot_comparison_for_preconditioning(
        self,
        preconditioning: int | Literal["identity", "inv_hess"],
        norm_variants: list[HessianNormalization] | None = None,
    ):
        if norm_variants is None:
            norm_variants = HessianNormalization.non_degenerate_choices()
        if isinstance(preconditioning, str):
            data = cast(
                pd.DataFrame,
                self.df[[col for col in self.df.columns if preconditioning in col]],
            )
        else:
            data = self.get_data_for_hess_iteration(preconditioning)

        def to_label(col: str):
            return (
                norm.to_plot_label()
                if (norm := extract_normalization_from_column(col)) is not None
                else "brak normalizacji"
            )

        data = data[
            [
                col
                for col in data.columns
                if any(
                    extract_normalization_from_column(col) == v for v in norm_variants
                )
            ]
        ]
        with self.new_ax() as ax:
            plot_with_legend_function(data, ax, to_label)  # pyright: ignore[reportArgumentType]

        if isinstance(preconditioning, int):
            prec_for_title = tex(f"H_{{{preconditioning}}}")
        elif preconditioning == "identity":
            prec_for_title = tex("I")
        else:
            prec_for_title = tex("H^{-1}")
        plt.title(f"d={self.dimensions}, $H_{{inv0}}$={prec_for_title}")

        if self.save_to_disk:
            plt.savefig(
                self.output_directory / f"by_norm_{preconditioning}.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def get_all_preconditioning_variants(self):
        seen = set()
        for cols in self.df.columns:
            if (iters := extract_iterations_from_column(cols)) is not None:
                seen.add(iters)
            elif "identity" in cols:
                seen.add("identity")
            elif "inv_hess" in cols:
                seen.add("inv_hess")

        return seen


def plot_all_random_x0(dim):
    plotter = BFGSAccelerationPlotter(
        dimensions=dim,
        x0_mode="random",
        manuscript_version=True,
        save_to_disk=True,
    )
    for norm in HessianNormalization:
        plotter.plot_comparison_for_norm(norm)

    for prec in plotter.get_all_preconditioning_variants():
        plotter.plot_comparison_for_preconditioning(prec)


if __name__ == "__main__":
    debug = bool(os.getenv("DEBUG", ""))
    print(f"Debug mode: {debug}")
    if debug:
        plotter = BFGSAccelerationPlotter(100, "inherited", save_to_disk=False)
    else:
        DIMS = [10, 20, 50, 100]

        Parallel(n_jobs=-1)(
            delayed(plot_all_random_x0)(dim)
            for dim in tqdm(DIMS, desc="Processing dimension*x0_mode sets...")
        )
