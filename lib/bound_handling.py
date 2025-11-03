from enum import Enum

import numpy as np
from loguru import logger


class RepairMethod(Enum):
    REFLECT = "reflect"


class OutOfBoundsError(Exception): ...


def repair_by_reflection(individual: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
    low, high = bounds
    repaired = individual.copy()

    below = repaired < low
    above = repaired > high
    needs_repair = np.any(below | above)
    if needs_repair:
        logger.warning(
            f"individual needs repair: {individual} out of bounds {bounds}", repaired
        )
    else:
        logger.info(
            f"individual doesn't need repair: {individual} in bounds {bounds}", repaired
        )
        return individual

    repaired[below] = low + (low - repaired[below])
    repaired[below] = np.clip(repaired[below], low, high)

    repaired[above] = high - (repaired[above] - high)
    repaired[above] = np.clip(repaired[above], low, high)
    logger.error(f"individual {individual} after repair: {repaired}")

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
