import os
import re
from dataclasses import dataclass, field
from itertools import product

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from experiments.find_switch_interval.cmabfgs.experiment_config import (
    CMABFGSExperimentConfig,
)
from experiments.find_switch_interval.common import (
    ObjectiveChoice,
    OptimumPosition,
)
from lib.enums import HessianNormalization
from lib.serde import aggregate_convergence_series
from lib.util import compress_and_save, summarize_data

ANY_INT = 0


@dataclass
class CMABFGSPostprocessor:
    config: CMABFGSExperimentConfig
    remove_outliers: bool = False
    multipliers: list[float] = field(
        init=False, default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0]
    )

    @property
    def input_file(self):
        return self.config.output_directory / "raw.parquet"

    @property
    def raw_curves_output_file(self):
        filename = (
            "raw_curves.parquet"
            if not self.remove_outliers
            else "raw_curves_no_outliers.parquet"
        )
        return self.config.output_directory / filename

    @property
    def agg_curves_output_file(self):
        filename = (
            "agg_curves.parquet"
            if not self.remove_outliers
            else "agg_curves_no_outliers.parquet"
        )
        return self.config.output_directory / filename

    @property
    def agg_cmaes_output_file(self):
        filename = (
            "agg.parquet" if not self.remove_outliers else "agg_no_outliers.parquet"
        )
        return self.config.input_file.parent / filename

    def is_divisible_by_multiplier(self, label: str, multiplier: float):
        iters = int(self.config.dimensions * multiplier)
        match = re.match(r"best_(\d+)", label)
        if match is None:
            return False
        return int(match.group(1)) % iters == 0

    @staticmethod
    def extract_iters_from_label(label: str):
        return int(label.lstrip("best_"))

    def get_curve_for_multiplier(
        self, data: pd.DataFrame, multiplier: float, spans: dict[str, int]
    ):
        bfgs_columns = [
            col
            for col in data.columns
            if self.is_divisible_by_multiplier(col, multiplier)
        ]
        cmaes_series = (
            data[["best_cmaes"]]
            .assign(iteration=data["best_cmaes"].index // (4 * self.config.dimensions))
            .dropna()
        )
        # offset cmaes evals
        offset = pd.Series(0, index=cmaes_series.index)
        for col in bfgs_columns:
            threshold = self.extract_iters_from_label(col)
            mask = cmaes_series["iteration"] > threshold
            offset.loc[mask] += spans[col]
        cmaes_series.index = cmaes_series.index + offset
        cmaes_series = cmaes_series.rename_axis("num_evaluations")

        bfgs_curves = []

        # offset bfgs evals
        for col in bfgs_columns:
            start_iter = self.extract_iters_from_label(col)

            start_eval = cmaes_series.loc[
                cmaes_series["iteration"] == start_iter
            ].index.min()

            s = data[col].dropna()
            s.index = (s.index + start_eval).rename("num_evaluations")
            bfgs_curves.append(s)

        series = [cmaes_series["best_cmaes"]] + [s for s in bfgs_curves if not s.empty]

        curve = pd.concat(series).sort_index().cummin().groupby(level=0).min()
        return curve

    def get_span_dict(self, data: pd.DataFrame):
        rv: dict[str, int] = {}
        for col in [c for c in data.columns if re.match(r"best_\d+", c)]:
            df = data[col].dropna()
            rv[col] = df.index.max() - df.index.min()  # pyright: ignore[reportOperatorIssue]
        return rv

    def run_subprocess(self, run_id: int, df: pd.DataFrame):
        spans = self.get_span_dict(df)
        results = []

        for mul in self.multipliers:
            curve = self.get_curve_for_multiplier(df, mul, spans)

            out = (
                curve.rename("value")
                .reset_index()
                .assign(run_id=run_id, multiplier=mul)
            )
            results.append(out)

        df_out = pd.concat(results, ignore_index=True)
        vanilla_bfgs_curve = (
            df["best_vanilla_bfgs"]  # pyright: ignore[reportCallIssue]
            .rename("value")  # pyright: ignore[reportArgumentType]
            .reset_index()
            .dropna()
            .assign(run_id=run_id, multiplier=0.0)
        )
        df_out = pd.concat(
            [df_out, vanilla_bfgs_curve],
            ignore_index=True,
        )
        df_out["multiplier"] = df_out["multiplier"].astype("category")
        return df_out

    def archive_data(
        self, raw: pd.DataFrame, agg: pd.DataFrame, agg_cmaes: pd.DataFrame
    ):
        compress_and_save(raw, self.raw_curves_output_file)
        if self.config.debug:
            df = pd.read_parquet(self.raw_curves_output_file)
            summarize_data(df)

        compress_and_save(agg, self.agg_curves_output_file)
        if self.config.debug:
            df = pd.read_parquet(self.agg_curves_output_file)
            summarize_data(df)

        compress_and_save(agg_cmaes, self.agg_cmaes_output_file)
        if self.config.debug:
            df = pd.read_parquet(self.agg_cmaes_output_file)
            summarize_data(df)

    def run(self, n_jobs: int = -1):
        try:
            raw = pd.read_parquet(self.input_file)
        except Exception:
            logger.error(f"MISSING RUN FILE FOR CONFIGURATION {self.config}")
            return
        cmaes_series = [df[["best_cmaes"]].dropna() for _, df in raw.groupby("run_id")]
        agg_cmaes = aggregate_convergence_series(cmaes_series, self.remove_outliers)  # pyright: ignore[reportArgumentType]

        grouped = list(raw.groupby("run_id"))

        dfs = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self.run_subprocess)(run_id, df)
            for run_id, df in tqdm(grouped, "Processing runs")
        )

        raw = pd.concat(dfs, ignore_index=True)  # pyright: ignore
        agg = self.aggregate_curves(raw)
        self.archive_data(raw, agg, agg_cmaes)

    def aggregate_curves(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []

        for mul, df_mul in df.groupby("multiplier"):
            curves = []
            grids = []

            for _, g in df_mul.groupby("run_id"):
                s = g.set_index("num_evaluations").sort_index()["value"]
                curves.append(s)
                grids.append(s.index)

            agg = (
                aggregate_convergence_series(curves, self.remove_outliers)
                .reset_index()
                .assign(multiplier=mul)
            )
            results.append(agg)

        return pd.concat(results, ignore_index=True)


def process_config(config: CMABFGSExperimentConfig):
    processor = CMABFGSPostprocessor(config, True)
    processor.run()


if __name__ == "__main__":
    debug = bool(os.getenv("DEBUG", ""))
    print(f"Debug mode: {debug}")

    if debug:
        config = CMABFGSExperimentConfig(
            10,
            ANY_INT,
            ObjectiveChoice.CEC1,
            OptimumPosition.MIDDLE,
            True,
            HessianNormalization.UNIT,
        )
        CMABFGSPostprocessor(config, remove_outliers=False).run()

    else:
        hess_norms = [
            HessianNormalization.UNIT_DIM,
            HessianNormalization.UNIT,
        ]

        # CEC configurations
        cec_optimum_positions = [
            OptimumPosition.MIDDLE,
        ]
        cec_dims = [10, 30, 50, 100]
        cec_objectives = [getattr(ObjectiveChoice, f"CEC{i}") for i in range(1, 31)]
        cec_configurations = [
            CMABFGSExperimentConfig(d, ANY_INT, obj, opt, False, hess_norm)
            for d, obj, opt, hess_norm in product(
                cec_dims, cec_objectives, cec_optimum_positions, hess_norms
            )
        ]

        # Control configurations
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
            delayed(process_config)(config) for config in all_configurations
        )
