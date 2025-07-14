from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd

from lib.metrics import Metric
from lib.util import EvalCounter


class HasCounter(Protocol):
    counter: EvalCounter


@dataclass
class MetricsCollector:
    metrics: Sequence[Metric]
    collect_method: str
    data: list = field(default_factory=list)

    def __call__(self, state: HasCounter):
        evals = state.counter.num_evaluations
        entry = {"num_evaluations": evals}
        for metric in self.metrics:
            collect_fn = getattr(metric, f"collect_{self.collect_method}")
            entry[metric.key()] = collect_fn(state)
        self.data.append(entry)

    def as_dataframe(self):
        return pd.DataFrame(self.data)

    def export_to_csv(self, path: Path):
        self.as_dataframe().to_csv(path, index=False)
