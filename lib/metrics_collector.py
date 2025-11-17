from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, cast

import pandas as pd

from lib.metrics import Metric
from lib.util import EvalCounter


class HasCounter(Protocol):
    counter: EvalCounter


@dataclass
class MetricsCollector:
    metrics: Sequence[Metric]
    collect_method: str
    run_id: int
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __call__(self, state: HasCounter, identifier: str = ""):
        evals = state.counter.num_evaluations

        entry = {"num_evaluations": [evals]}
        for metric in self.metrics:
            key = f"{metric.key()}_{identifier}" if identifier else metric.key()
            entry[key] = metric.collect(state)  # pyright: ignore[reportArgumentType]

        entry_df = pd.DataFrame(entry)
        if self.data.empty:
            self.data = entry_df
        else:
            self.data = pd.concat([self.data, entry_df])

    def validate(self):
        if not (self.data >= 0).all().all():
            raise ValueError("MetricsCollector contains negative values.")

    def as_dataframe(self):
        # squash entries with duplicate indices
        df = cast(pd.DataFrame, self.data)
        df = self.data.groupby(["num_evaluations"]).max()
        df["run_id"] = self.run_id
        return df

    def export_to_csv(self, path: Path):
        self.as_dataframe().to_csv(path)
