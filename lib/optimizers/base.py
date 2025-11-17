from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from lib.metrics_collector import MetricsCollector


class Optimizer(ABC):
    """Common interface for all optimizers"""

    @abstractmethod
    @abstractmethod
    def optimize(self, objective: Callable, callback: "MetricsCollector"): ...
