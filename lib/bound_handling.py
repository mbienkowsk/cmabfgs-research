from enum import Enum

import numpy as np


class RepairMethod(Enum):
    REFLECT = "reflect"


class OutOfBoundsError(Exception): ...


def repair_by_reflection(individual: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
    low, high = bounds
    repaired = individual.copy()

    below = repaired < low
    repaired[below] = np.clip(repaired[below], low, high)

    above = repaired > high
    repaired[above] = high - (repaired[above] - high)
    repaired[above] = np.clip(repaired[above], low, high)

    return repaired


def check_bounds(
    individual: np.ndarray, bounds: tuple[int, int], raise_exception: bool = True
) -> bool:
    low, high = bounds
    good = bool(np.all((individual >= low) & (individual <= high)))
    if good:
        return True
    if raise_exception:
        raise OutOfBoundsError(f"Individual {individual} is out of bounds {bounds}.")
    return False
