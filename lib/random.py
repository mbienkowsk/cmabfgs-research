from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.random import Generator
from sympy import prime


@dataclass
class IndividualGenerator:
    """Wrapper around np.random.Generator. seed is set to run_id-th prime number."""

    run_id: int
    bounds: tuple[float, float]
    dimensions: int

    seed: int = field(init=False)
    rng: Generator = field(init=False)

    def __post_init__(self):
        self.seed = cast(int, prime(self.run_id))
        self.rng = np.random.default_rng(self.seed)

    def get_n_individuals(self, n: int):
        return self.rng.uniform(
            low=self.bounds[0], high=self.bounds[1], size=(n, self.dimensions)
        ).reshape((n, self.dimensions))

    def get_individual(self):
        return self.get_n_individuals(1)[0]
