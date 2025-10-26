from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from cmaes import CMA
from loguru import logger
from scipy.optimize import OptimizeResult, minimize

from lib.callbacks import HasCounter
from lib.optimizers.base import Optimizer
from lib.optimizers.bfgs import BFGSState
from lib.optimizers.cmaes import CMAESState
from lib.stopping import (BFBGSEarlyStopping, CMAESEarlyStopping,
                          StopOptimization)
from lib.util import EvalCounter

if TYPE_CHECKING:
    from lib.callbacks import MetricsCollector


VANILLA = "vanilla_cmaes"


@dataclass
class MultiCMABFGSState:
    nums_cmaes_evaluations: list[int]
    mode: Literal["CMAES", "BFGS"]
    cmaes_state: CMAESState
    bfgs_state: BFGSState
    key_suffix: str = field(default=VANILLA)

    @property
    def num_cmaes_evaluations(self) -> int:
        return self.cmaes_state.num_evaluations

    @property
    def counter(self) -> EvalCounter:
        match self.mode:
            case "CMAES":
                return self.cmaes_state.counter
            case "BFGS":
                return self.bfgs_state.counter

    @property
    def bfgs_counter(self):
        return self.bfgs_state.counter

    @bfgs_counter.setter
    def bfgs_counter(self, value):
        self.bfgs_state.counter = value

    @property
    def cmaes_counter(self):
        return self.cmaes_state.counter

    @cmaes_counter.setter
    def cmaes_counter(self, value):
        self.cmaes_state.counter = value


class MultiCMABFGS(Optimizer):
    state: MultiCMABFGSState

    def __init__(
        self,
        x0: np.ndarray,
        n_cmaes_iterations: list[int],
        seed: int,
        fun: EvalCounter,
        popsize: int,
        callback: "MetricsCollector",
        cmaes_stopper: CMAESEarlyStopping,
        sigma: int = 1,
    ):
        self.cma = CMA(
            mean=x0,
            sigma=sigma,
            seed=seed,
            population_size=popsize,
        )
        self.seed = seed
        self.nums_cmaes_iterations = n_cmaes_iterations
        self.callback = callback
        self.cmaes_stopper = cmaes_stopper
        cmaes_state = CMAESState(
            fun,
            self.cma._C,
        )
        self.bfgs_stopper = BFBGSEarlyStopping(max_evals=self.cmaes_stopper.max_evals)
        bfgs_state = BFGSState(fun)
        self.state = MultiCMABFGSState(
            n_cmaes_iterations, "CMAES", cmaes_state, bfgs_state
        )

    def cma_step(self):
        solutions = []

        for _ in range(self.cma.population_size):
            x = self.cma.ask()
            solutions.append((x, self.state.counter(x)))

        self.cma.tell(solutions)

        return solutions.copy()

    def switch_to_cmaes(self):
        self.state.key_suffix = VANILLA
        self.state.mode = "CMAES"

    def switch_to_bfgs(self, suffix: str):
        self.state.mode = "BFGS"
        self.state.key_suffix = suffix
        self.state.bfgs_counter = deepcopy(self.state.cmaes_counter)

    def finish_cmaes(self):
        self.switch_to_cmaes()
        while not self.cmaes_stopper:
            self.cma_step()
            self.callback(cast(HasCounter, self.state))

    def optimize(self):
        shifted = [0] + self.nums_cmaes_iterations[:-1]
        differences = [x - y for x, y in zip(self.nums_cmaes_iterations, shifted)]
        for idx, switch_after in enumerate(differences):
            self.switch_to_cmaes()
            for _ in range(switch_after):
                self.cma_step()
                self.callback(cast(HasCounter, self.state))

            self.switch_to_bfgs(str(self.nums_cmaes_iterations[idx]))
            self.callback(cast(HasCounter, self.state))

            def callback_wrapper(intermediate_result: OptimizeResult):
                self.bfgs_stopper(self.state.bfgs_state)  # raises an exception
                self.state.bfgs_state.current_result = intermediate_result
                self.callback(cast(HasCounter, self.state))

            try:
                minimize(
                    self.state.counter,
                    self.cma.mean,
                    method="BFGS",
                    callback=callback_wrapper,
                    options={
                        "gtol": 1e-30,
                        "hess_inv0": (self.cma._C + self.cma._C.T) / 2,
                        # TODO: why is this still happening? investigate
                    },
                )
            except StopOptimization as e:
                logger.info(f"BFGS stopped early: {e}")

        self.finish_cmaes()
