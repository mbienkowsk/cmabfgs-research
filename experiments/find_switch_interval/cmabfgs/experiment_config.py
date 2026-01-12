import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

from experiments.find_switch_interval.common import (
    ExperimentConfigBase,
)
from lib.enums import HessianNormalization
from lib.util import evaluation_budget


@dataclass
class CMABFGSExperimentConfig(ExperimentConfigBase):
    max_evals: int = field(init=False)
    hess_normalization: HessianNormalization = HessianNormalization.UNIT_DIM

    def __post_init__(self):
        super().__post_init__()
        self.max_evals = evaluation_budget(self.dimensions)

    @property
    def debug_filename_stub(self):
        return (
            Path(__file__).parent
            / "debug"
            / f"{self.objective_choice.value}_d{self.dimensions}_debug"
        )

    @property
    @override
    def output_directory(self):
        return (
            Path(__file__).parent
            / "results"
            / self.objective_choice.value
            / str(self.dimensions)
            / self.optimum_position.value
            / self.hess_normalization.value
        )

    @property
    def input_file(self):
        return (
            Path(__file__).parent.parent
            / "cmaes"
            / "results"
            / self.objective_choice.value
            / str(self.dimensions)
            / self.optimum_position.value
            / "raw.parquet"
        )

    @override
    @classmethod
    def create_from_env(cls):
        base = super().create_from_env()
        return cls(
            dimensions=base.dimensions,
            num_runs=base.num_runs,
            objective_choice=base.objective_choice,
            optimum_position=base.optimum_position,
            debug=base.debug,
            hess_normalization=HessianNormalization(os.environ["HESS_NORM"]),
        )
