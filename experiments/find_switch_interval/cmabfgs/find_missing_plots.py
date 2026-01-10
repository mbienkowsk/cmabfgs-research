"""Script for finding missing CEC plots that errored out when generating for one reason or another"""

from itertools import product

from tqdm import tqdm

from experiments.find_switch_interval.cmabfgs.experiment_config import (
    CMABFGSExperimentConfig,
)
from experiments.find_switch_interval.cmabfgs.postprocessing import CMABFGSPostprocessor
from experiments.find_switch_interval.common import (
    ObjectiveChoice,
    OptimumPosition,
)
from lib.enums import HessianNormalization

ANY_INT = 0

if __name__ == "__main__":
    ALL_DIMS = [10, 30, 50, 100]
    ALL_CEC_OBJECTIVES = [getattr(ObjectiveChoice, f"CEC{i}") for i in range(1, 31)]

    for dim, obj in tqdm(
        product(ALL_DIMS, ALL_CEC_OBJECTIVES),
        total=len(ALL_DIMS) * len(ALL_CEC_OBJECTIVES),
    ):
        config = CMABFGSExperimentConfig(
            dim,
            ANY_INT,
            obj,
            OptimumPosition.MIDDLE,
            False,
            HessianNormalization.UNIT_DIM,
        )
        processor = CMABFGSPostprocessor(config)
        if not processor.plot_save_path.exists():
            print(f"Plot missing for configuration {config}")
