from enum import Enum

import numpy as np
from loguru import logger


class BoundEnforcement(Enum):
    ALLOW_OOB = "allow_oob"
    ADDITIVE_PENALTY = "additive_penalty"
    DEATH_PENALTY = "death_penalty"
    IGNORE_SOLUTIONS = "ignore_solutions"


class RepairMethod(Enum):
    REFLECT = "reflect"


class OutOfBoundsError(Exception): ...


def repair_by_reflection(
    individual: np.ndarray, bounds: tuple[float, float]
) -> np.ndarray:
    low, high = bounds
    repaired = individual.copy()

    below = repaired < low
    above = repaired > high
    needs_repair = np.any(below | above)
    if needs_repair:
        logger.debug(
            f"individual needs repair: {individual} out of bounds {bounds}", repaired
        )
    else:
        return individual

    repaired[below] = low + (low - repaired[below])
    repaired[below] = np.clip(repaired[below], low, high)

    repaired[above] = high - (repaired[above] - high)
    repaired[above] = np.clip(repaired[above], low, high)
    logger.debug(f"individual {individual} after repair: {repaired}")

    return repaired


def check_bounds(
    individual: np.ndarray, bounds: tuple[float, float], raise_exception: bool = True
) -> bool:
    low, high = bounds
    good = bool(np.all((individual >= low) & (individual <= high)))
    if good:
        return True
    if raise_exception:
        raise OutOfBoundsError(f"Individual {individual} is out of bounds {bounds}.")
    return False


def bound_dist_sq(individual: np.ndarray, bounds: tuple[float, float]) -> float:
    low, high = bounds
    penalty = 0.0
    below = individual < low
    above = individual > high
    if np.any(below):
        penalty += np.sum((low - individual[below]) ** 2)
    if np.any(above):
        penalty += np.sum((individual[above] - high) ** 2)
    return penalty
