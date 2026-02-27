import cocoex
import hydra
from omegaconf import DictConfig

from lib.optimizers.cmaes import CMAES
from lib.stopping import CMAESEarlyStopping
from lib.util import evaluation_budget

SUITE = "bbob"


def suite_for_config(cfg: DictConfig):
    return cocoex.Suite(SUITE, "", f"dimensions: {cfg['dimensions']}")


def optimizer_for_config(cfg: DictConfig):
    match cfg["optimizer"]:
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

        case "cmabfgs":

            def fmin(problem, x0): ...


@hydra.main(version_base=None, config_path="conf", config_name="cmaes")
def main(cfg: DictConfig) -> None:
    suite = suite_for_config(cfg)

    print(cfg)


if __name__ == "__main__":
    # fmin = scipy.optimize.fmin
    # suite = cocoex.Suite(SUITE, "", "")
    # observer = cocoex.Observer(SUITE, "")
    # batcher = cocoex.BatchScheduler(number_of_batches, batch_to_execute)
    #
    # for problem in suite:
    #     print(problem)
    #
    # print(len(suite))
    main()
