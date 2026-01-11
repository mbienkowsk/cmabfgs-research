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


def filter_for_normalization_method(df: pd.DataFrame, norm: HessianNormalization):
    return cast(
        pd.DataFrame,
        df[[col for col in df.columns if col.endswith(norm.value)]],
    )


Preconditioning = int | Literal["identity", "inv_hess"]


def preconditioning_label(preconditioning: Preconditioning) -> str:
    if isinstance(preconditioning, int):
        return tex(f"H_{{{preconditioning}}}")
    elif preconditioning == "identity":
        return tex("I")
    else:
        return tex("H^{-1}")


def filter_for_preconditioning(df: pd.DataFrame, prec: Preconditioning) -> pd.DataFrame:
    if isinstance(prec, int):
        cols = [
            col for col in df.columns if extract_iterations_from_column(col) == prec
        ]
        if not cols:
            all_iters = set(extract_iterations_from_column(col) for col in df)
            raise ValueError(
                f"No BFGS preconditioned with the matrix for input {prec}. Possible values: {all_iters}"
            )
        return cast(pd.DataFrame, df[cols])
    else:
        return df[[col for col in df.columns if prec in col]]  # pyright: ignore[reportReturnType]


def remove_non_cmaes_preconditioning_variants(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [col for col in df.columns if "identity" not in col and "inv_hess" not in col]
    ]  # pyright: ignore[reportReturnType]


@dataclass
class BFGSAccelerationPlotter:
    dimensions: int
    x0_mode: Literal["random", "inherited"]
    save_to_disk: bool = True

    def __post_init__(self):
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

    def finalize_plot(self, base_filename: str, filename_suffix: str):
        """Resolve filename and save/show based on the configuration"""
        if self.save_to_disk:
            filename = base_filename
            if filename_suffix:
                filename += f"_{filename_suffix}"
            plt.savefig(
                self.output_directory / f"{filename}.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.show()
        plt.close()

    def plot_comparison_for_norm(
        self, norm: HessianNormalization, data: pd.DataFrame, filename_suffix: str = ""
    ):
        def to_label(col: str):
            if (iters := extract_iterations_from_column(col)) is not None:
                return str(iters)
            return "I" if "identity" in col else "H_inv" if "inv_hess" in col else col

        with self.new_ax() as ax:
            plot_with_legend_function(data, ax, to_label)
            plt.title(norm.to_plot_label())

        self.finalize_plot(f"by_iteration_{norm.value}", filename_suffix)

    def plot_comparison_for_preconditioning(
        self,
        df: pd.DataFrame,
        preconditioning: int | Literal["identity", "inv_hess"],
        norm_variants: list[HessianNormalization] | None = None,
        filename_suffix: str = "",
    ):
        if norm_variants is None:
            norm_variants = HessianNormalization.non_degenerate_choices()

        data = filter_for_preconditioning(df, preconditioning)

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

        plt.title(
            f"$d={self.dimensions}$, $H_{{inv0}}$={preconditioning_label(preconditioning)}"
        )

        self.finalize_plot(f"by_norm_{preconditioning}", filename_suffix)

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

    def plot_comparison_for_preconditioning_from_starting_point(
        self,
        data: pd.DataFrame,
        starting_point: int,
        norm: HessianNormalization,
    ):
        def to_label(col: str):
            if "identity" in col:
                return tex("I")
            elif "inv_hess" in col:
                return tex("H_{inv}")
            return tex("C_{" + str(extract_iterations_from_column(col)) + "}")

        data = filter_for_preconditioning(data, starting_point)
        with self.new_ax() as ax:
            plot_with_legend_function(data, ax, to_label)

        plt.title(
            f"$d={self.dimensions}$, $x_0=m_{{{starting_point}}}$, {norm.to_plot_label()}"
        )

        self.finalize_plot(f"from_position_{starting_point}_{norm.value}", "")


def plot_all_random_x0(dim):
    plotter = BFGSAccelerationPlotter(
        dimensions=dim,
        x0_mode="random",
        save_to_disk=True,
    )
    for norm in HessianNormalization:
        data = filter_for_normalization_method(plotter.df, norm)
        plotter.plot_comparison_for_norm(norm, data)
    for prec in plotter.get_all_preconditioning_variants():
        data = filter_for_preconditioning(plotter.df, prec)
        plotter.plot_comparison_for_preconditioning(data, prec)


def plot_all_inherited_x0(dim):
    plotter = BFGSAccelerationPlotter(
        dimensions=dim,
        x0_mode="inherited",
        save_to_disk=True,
    )
    for norm in HessianNormalization:
        data = filter_for_normalization_method(plotter.df, norm)
        data = remove_non_cmaes_preconditioning_variants(data)
        plotter.plot_comparison_for_norm(
            norm, data, filename_suffix="only_cmaes_preconditioning"
        )
    for prec in plotter.get_all_preconditioning_variants():
        data = filter_for_preconditioning(plotter.df, prec)
        data = remove_non_cmaes_preconditioning_variants(data)
        plotter.plot_comparison_for_preconditioning(
            data, prec, filename_suffix="only_cmaes_preconditioning"
        )
    for prec in plotter.get_all_preconditioning_variants():
        if isinstance(prec, int):
            data = filter_for_preconditioning(plotter.df, prec)
            norm = HessianNormalization.UNIT
            data = filter_for_normalization_method(data, norm)

            plotter.plot_comparison_for_preconditioning_from_starting_point(
                data, prec, norm
            )


if __name__ == "__main__":
    debug = bool(os.getenv("DEBUG", ""))
    print(f"Debug mode: {debug}")
    if debug:
        plotter = BFGSAccelerationPlotter(100, "inherited", save_to_disk=False)
        # plotter.plot_comparison_for_norm(HessianNormalization.UNIT)
    else:
        DIMS = [10, 20, 50, 100]

        Parallel(n_jobs=-1)(
            delayed(plot_all_random_x0)(dim)
            for dim in tqdm(
                DIMS, desc="Processing dimension*x0_mode sets for random x0..."
            )
        )
        Parallel(n_jobs=-1)(
            delayed(plot_all_inherited_x0)(dim)
            for dim in tqdm(
                DIMS, desc="Processing dimension*x0_mode sets for inherited x0..."
            )
        )
