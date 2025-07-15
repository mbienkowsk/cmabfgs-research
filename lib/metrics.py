from abc import ABC, abstractmethod
from typing import Any

from lib.optimizers.bfgs import BFGSState
from lib.optimizers.cmabfgs import CMABFGSState, CMAESState
from lib.optimizers.cmaes import CMAESState


class Metric(ABC):
    @abstractmethod
    def key(self) -> str: ...

    def collect_cmaes(self, state: CMAESState) -> Any:
        raise NotImplementedError

    def collect_bfgs(self, state: BFGSState) -> Any:
        raise NotImplementedError

    def collect_cmabfgs(self, state: CMABFGSState) -> Any:
        if state.mode == "CMAES":
            return self.collect_cmaes(state.cmaes_state)
        elif state.mode == "BFGS":
            return self.collect_bfgs(state.bfgs_state)


class MeanEvaluation(Metric):
    def key(self):
        return "mean"

    def collect_cmaes(self, state: CMAESState):
        return state.counter.without_counting(state.mean)

    def collect_cmabfgs(self, state: CMABFGSState):
        if state.mode == "CMAES":
            return self.collect_cmaes(state.cmaes_state)
        return None


class BestSoFar(Metric):
    def key(self):
        return "best"

    def collect_cmaes(self, state: CMAESState):
        return state.best_solutions[-1] if state.best_solutions else None

    def collect_bfgs(self, state: BFGSState):
        return state.best_solutions[-1] if state.best_solutions else None
