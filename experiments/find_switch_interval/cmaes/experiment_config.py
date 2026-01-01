import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from loguru import logger

from lib.funs import get_function_by_name


class ObjectiveChoice(Enum):
    ELLIPTIC = "Elliptic"
    RASTRIGIN = "Rastrigin"


class OptimumPosition(Enum):
    MIDDLE = "middle"
    CORNER = "corner"
    OUTSIDE_CORNER = "outside_corner"

    def get_bounds(self):
        match self:
            case OptimumPosition.MIDDLE:
                return (-100.0, 100.0)
            case OptimumPosition.CORNER:
                return (-180.0, 20.0)
            case OptimumPosition.OUTSIDE_CORNER:
                return (-220.0, -20.0)

    def to_plot_label(self):
        bounds = self.get_bounds()
        match self:
            case OptimumPosition.MIDDLE:
                label = "pośrodku obszaru dopuszczalnego"
            case OptimumPosition.CORNER:
                label = "w rogu obszaru dopuszczalnego"
            case OptimumPosition.OUTSIDE_CORNER:
                label = "poza rogiem obszaru dopuszczalnego"
        return f"{label} (granice: {bounds})"


@dataclass
class CMAESExperimentConfig:
    dimensions: int
    num_runs: int
    objective_choice: ObjectiveChoice
    optimum_position: OptimumPosition
    debug: bool = False

    max_evals: int = field(init=False)
    popsize: int = field(init=False)
    collection_interval: int = field(init=False)
    bounds: tuple[float, float] = field(init=False)

    def __post_init__(self):
        self.max_evals = 10_000 * self.dimensions
        self.popsize = 4 * self.dimensions
        self.collection_interval = self.dimensions // 2
        self.bounds = self.optimum_position.get_bounds()
        self.output_directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_from_env(cls):
        try:
            return cls(
                dimensions=int(os.environ["DIMENSIONS"]),
                num_runs=int(os.environ["N_RUNS"]),
                objective_choice=ObjectiveChoice(os.environ["OBJECTIVE_CHOICE"]),
                optimum_position=OptimumPosition(
                    os.environ["OPTIMUM_POSITION"],
                ),
            )
        except KeyError as e:
            logger.error(f"Environment variable {e} is not set.")
            raise e

    def get_objective_instance(self):
        return get_function_by_name(self.objective_choice.value, self.dimensions)

    def get_run_indices(self):
        return range(1, self.num_runs)

    @property
    def output_directory(self):
        return (
            Path(__file__).parent
            / "results"
            / self.objective_choice.value
            / str(self.dimensions)
            / self.optimum_position.value
        )
