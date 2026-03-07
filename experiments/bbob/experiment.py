import cocoex
import hydra
from loguru import logger
from omegaconf import DictConfig

from lib.optimizers.cmaes import CMAES
from lib.optimizers.hybrids.multicmabfgs import MultiCMABFGS
from lib.stopping import CMAESEarlyStopping
from lib.util import EvalCounter, evaluation_budget, hansen_cmaes_popsize

SUITE = "bbob"


def suite_for_config(cfg: DictConfig):
    return cocoex.Suite(SUITE, "", f"dimensions: {cfg['dimensions']}")


def algorithm_name(cfg: DictConfig):
    core = (
        "cmaes"
        if ((name := cfg["optimizer"]) == "cmaes")
        else f"{name}_k-{cfg['optimizers']['cmabfgs']['k']}_precon-{cfg['optimizers']['cmabfgs']['precondition']}_scal-{cfg['optimizers']['cmabfgs']['scaling']}"
    )
    return f"{core}_pops-{cfg['popsize']}_stop_{cfg['stopping_criterion']}"


def observer_for_config(cfg: DictConfig):
    return cocoex.Observer(
        SUITE,
        f"outer_folder: {result_folder_for_config(cfg)} algorithm_name: {algorithm_name(cfg)}",
    )


def batcher_for_config(cfg: DictConfig):
    return cocoex.BatchScheduler(cfg["n_batches"], cfg["batch"])


def result_folder_for_config(cfg: DictConfig):
    return f"experiments/bbob/results/{algorithm_name(cfg)}"


def optimizer_for_config(cfg: DictConfig):
    dimensions = cfg["dimensions"]

    match cfg["popsize"]:
        case "hansen":
            popsize = hansen_cmaes_popsize(dimensions)
        case "beyer":
            popsize = 4 * dimensions

    match cfg["optimizer"]:
        case "cmaes":

            def fmin(problem, x0):
                cma = CMAES(
                    problem,
                    x0,
                    popsize,
                    cfg["seed"],
                    CMAESEarlyStopping(evaluation_budget(dimensions)),
                    [],
                    (-5.0, 5.0),
                )
                return cma.optimize()

            return fmin

        case "cmabfgs":
            k = cfg["optimizers"]["cmabfgs"]["k"]
            precondition = cfg["optimizers"]["cmabfgs"]["precondition"]

            def fmin(problem, x0):
                def switch_to_bfgs_oracle(cmaes_state, cmaes_iters_done):
                    return cmaes_iters_done % (k * dimensions) == 0

                cmabfgs = MultiCMABFGS(
                    x0,
                    switch_to_bfgs_oracle,
                    cfg["seed"],
                    problem,
                    popsize,
                    [],
                    CMAESEarlyStopping(evaluation_budget(dimensions)),
                    evaluation_budget(dimensions),
                    (-5, 5),
                    precondition_using_C=precondition,
                )
                return cmabfgs.optimize()

            return fmin

        case _:
            raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
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
