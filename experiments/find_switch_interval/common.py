import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
from loguru import logger

from lib.funs import get_function_by_name


class ObjectiveChoice(Enum):
    ELLIPTIC = "Elliptic"
    RASTRIGIN = "Rastrigin"
    CEC1 = "CEC_1"
    CEC2 = "CEC_2"
    CEC3 = "CEC_3"
    CEC4 = "CEC_4"
    CEC5 = "CEC_5"
    CEC6 = "CEC_6"
    CEC7 = "CEC_7"
    CEC8 = "CEC_8"
    CEC9 = "CEC_9"
    CEC10 = "CEC_10"
    CEC11 = "CEC_11"
    CEC12 = "CEC_12"
    CEC13 = "CEC_13"
    CEC14 = "CEC_14"
    CEC15 = "CEC_15"
    CEC16 = "CEC_16"
    CEC17 = "CEC_17"
    CEC18 = "CEC_18"
    CEC19 = "CEC_19"
    CEC20 = "CEC_20"
    CEC21 = "CEC_21"
    CEC22 = "CEC_22"
    CEC23 = "CEC_23"
    CEC24 = "CEC_24"
    CEC25 = "CEC_25"
    CEC26 = "CEC_26"
    CEC27 = "CEC_27"
    CEC28 = "CEC_28"
    CEC29 = "CEC_29"
    CEC30 = "CEC_30"


class HessianNormalization(Enum):
    UNIT = "unit"
    UNIT_DIM = "unit_dim"

    def to_plot_label(self):
        match self:
            case HessianNormalization.UNIT:
                return "skalowanie jednostkowe ($sqrt{sum_{ij}{C_{ij}^2}} = 1$)"
            case HessianNormalization.UNIT_DIM:
                return "skalowanie jednostkowe do wymiarowości ($sqrt{sum{ij}{C_{ij}^2}} = d$)"


class OptimumPosition(Enum):
    MIDDLE = "middle"
    CORNER = "corner"
    OUTSIDE_CORNER = "outside_corner"
    CORNER_NEAR = "corner_near"

    def get_bounds(self):
        match self:
            case OptimumPosition.MIDDLE:
                return (-100.0, 100.0)
            case OptimumPosition.CORNER:
                return (-180.0, 20.0)
            case OptimumPosition.OUTSIDE_CORNER:
                return (-220.0, -20.0)
            case OptimumPosition.CORNER_NEAR:
                return (-199.0, 1.0)

    def to_plot_label(self):
        bounds = self.get_bounds()
        match self:
            case OptimumPosition.MIDDLE:
                label = "pośrodku obszaru dopuszczalnego"
            case OptimumPosition.CORNER:
                label = "w rogu obszaru dopuszczalnego"
            case OptimumPosition.OUTSIDE_CORNER:
                label = "poza rogiem obszaru dopuszczalnego"
            case OptimumPosition.CORNER_NEAR:
                label = "w samym rogu obszaru dopuszczalnego"
        return f"{label} (granice: {bounds})"


@dataclass
class ExperimentConfigBase:
    dimensions: int
    num_runs: int
    objective_choice: ObjectiveChoice
    optimum_position: OptimumPosition
    debug: bool
    bounds: tuple[float, float] = field(init=False)

    def __post_init__(self):
        self.bounds = self.optimum_position.get_bounds()
        self.output_directory.mkdir(parents=True, exist_ok=True)

    @property
    def output_directory(self):
        raise NotImplementedError()

    def get_run_indices(self):
        return range(1, self.num_runs + 1)

    def get_objective_instance(self):
        return get_function_by_name(self.objective_choice.value, self.dimensions)

    @classmethod
    def create_from_env(cls):
        """Since this is used on the cluster, debug is hardcoded
        as False"""
        try:
            return cls(
                dimensions=int(os.environ["DIMENSIONS"]),
                num_runs=int(os.environ["N_RUNS"]),
                objective_choice=ObjectiveChoice(os.environ["OBJECTIVE_CHOICE"]),
                optimum_position=OptimumPosition(
                    os.environ["OPTIMUM_POSITION"],
                ),
                debug=False,
            )
        except KeyError as e:
            logger.error(f"Environment variable {e} is not set.")
            raise e


@dataclass
class ExperimentBase[ConfigType: ExperimentConfigBase](ABC):
    config: ConfigType

    @abstractmethod
    def run_subprocess(self, run_id) -> pd.DataFrame: ...

    @abstractmethod
    def archive_data(self, data: list[pd.DataFrame]): ...

    def run(self):
        with mp.Pool(mp.cpu_count()) as pool:
            run_indices = self.config.get_run_indices()
            data = pool.map(self.run_subprocess, run_indices)
            self.archive_data(data)
