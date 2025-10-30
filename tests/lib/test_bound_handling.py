import numpy as np
import pytest

from lib.bound_handling import (OutOfBoundsError, check_bounds,
                                repair_by_reflection)

bounds_tests_instances = [
    ((-10, 10), np.array([-10.0, 0.5, 9.9]), True, np.array([-10.0, 0.5, 9.9])),
    ((-10, 10), np.array([10.0, -10.0, 0.0]), True, np.array([10.0, -10.0, 0.0])),
    ((-10, 10), np.array([-11.5, 0.0, 5.5]), False, np.array([-8.5, 0.0, 5.5])),
    ((-10, 10), np.array([12.3, -9.9, 10.1]), False, np.array([7.7, -9.9, 9.9])),
    ((-10, 10), np.array([-15.0, 0.0, 30.0]), False, np.array([-5.0, 0.0, -10])),
]


@pytest.mark.parametrize(
    "bounds,individual,expected,repaired_by_reflection", bounds_tests_instances
)
def test_check_bounds(bounds, individual, expected, repaired_by_reflection):
    if not expected:
        with pytest.raises(OutOfBoundsError):
            assert check_bounds(individual, bounds) == expected
    else:
        check_bounds(individual, bounds)


@pytest.mark.parametrize(
    "bounds,individual,expected,repaired_by_reflection", bounds_tests_instances
)
def test_repair_by_reflection(bounds, individual, expected, repaired_by_reflection):
    assert np.allclose(repair_by_reflection(individual, bounds), repaired_by_reflection)
