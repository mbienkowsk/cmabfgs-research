from dataclasses import dataclass, field
from pathlib import Path
from typing import override

from experiments.find_switch_interval.common import (
    ExperimentConfigBase,
)
from lib.util import evaluation_budget


@dataclass
class CMAESExperimentConfig(ExperimentConfigBase):
    max_evals: int = field(init=False)
    popsize: int = field(init=False)
    collection_interval: int = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.max_evals = evaluation_budget(self.dimensions)
        self.popsize = 4 * self.dimensions
        self.collection_interval = self.dimensions // 2

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
