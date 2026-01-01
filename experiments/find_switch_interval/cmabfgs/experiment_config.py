from dataclasses import dataclass, field
from pathlib import Path
from typing import override

from experiments.find_switch_interval.common import (
    ExperimentConfigBase,
)

# TODO: ubic bfgs jak linesearch wyjdzie poza granice


@dataclass
class CMABFGSExperimentConfig(ExperimentConfigBase):
    max_evals: int = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.max_evals = 10_000 * self.dimensions

    @property
    @override
    def output_directory(self):
        return (
            Path(__file__).parent
            / "results"
            / self.objective_choice.value
            / str(self.dimensions)
            / self.optimum_position.value
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
