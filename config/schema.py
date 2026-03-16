from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from experiments.cov_mat_scaling_analysis.c_scale_convergence.experiment import (
    CScaleConvergenceExperimentConfig,
)


@dataclass
class MasterConfig:
    experiments: dict[str, DictConfig]


def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="base_config", node=MasterConfig)
    cs.store(
        group="experiments",
        name="c_scale_inference",
        node=CScaleConvergenceExperimentConfig,
    )
