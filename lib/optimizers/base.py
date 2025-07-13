from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from lib.callbacks import ExperimentCallback


class Optimizer(ABC):
    """Common interface for all optimizers"""

    state: Any

    @abstractmethod
    def step(self, objective: Callable): ...

    @abstractmethod
    @abstractmethod
    def optimize(self, objective: Callable, callback: "ExperimentCallback"): ...
