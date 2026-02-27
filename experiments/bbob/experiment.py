from pathlib import Path

import cocoex
import hydra
from loguru import logger
from omegaconf import DictConfig

from lib.optimizers.cmaes import CMAES
from lib.stopping import CMAESEarlyStopping
from lib.util import EvalCounter, evaluation_budget

SUITE = "bbob"


def suite_for_config(cfg: DictConfig):
    return cocoex.Suite(SUITE, "", f"dimensions: {cfg['dimensions']}")


def algorithm_name(cfg: DictConfig):
    return (
        "cmaes"
        if (name := cfg["optimizer"]["name"] == "cmaes")
        else f"{name}-{cfg['optimizer']['k']}"
    )


def observer_for_config(cfg: DictConfig):
    return cocoex.Observer(
        SUITE,
        f"outer_folder: {result_folder_for_config(cfg)} algorithm_name: {algorithm_name(cfg)}",
    )


def batcher_for_config(cfg: DictConfig):
    return cocoex.BatchScheduler(cfg["n_batches"], cfg["batch"])


def result_folder_for_config(cfg: DictConfig):
    return Path(__file__).parent / f"results/{cfg['optimizer']['name']}"


def optimizer_for_config(cfg: DictConfig):
    match cfg["optimizer"]["name"]:
        case "cmaes":

            def fmin(problem, x0):
                cma = CMAES(
                    problem,
                    x0,
                    4 * cfg["dimensions"],
                    cfg["seed"],
                    CMAESEarlyStopping(evaluation_budget(cfg["dimensions"])),
                    [],
                    (-5.0, 5.0),
                )
                return cma.optimize()

            return fmin
        case _:
            raise NotImplementedError("Not done yet")


@hydra.main(version_base=None, config_path="conf", config_name="cmaes")
def main(cfg: DictConfig) -> None:
    suite = suite_for_config(cfg)
    logger.info(f"Created suite with {len(suite)} problems")

    optimizer = optimizer_for_config(cfg)
    observer = observer_for_config(cfg)
    batcher = batcher_for_config(cfg)

    for problem in suite:
        logger.info(f"Starting problem {problem}")
        if batcher.is_in_batch(problem):
            problem.observe_with(observer)
            try:
                optimizer(EvalCounter(problem), problem.initial_solution)
            except Exception as e:
                logger.error(f"Error optimizing problem {problem}: {e}")


if __name__ == "__main__":
    main()
