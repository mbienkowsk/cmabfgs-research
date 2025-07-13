import numpy as np
from scipy.optimize import minimize

from lib.callbacks import ExperimentCallback
from lib.optimizers.base import Optimizer
from lib.util import EvalCounter


class LBFGS(Optimizer):
    def __init__(self, x0: np.ndarray, seed: int, fun: EvalCounter):
        self.x0 = x0
        self.inner = None
        self.seed = seed
        self.fun = fun

    def optimize(self, callback: ExperimentCallback):
        minimize(
            self.fun,
            self.x0,
            method="L-BFGS-B",
            callback=callback,
        )
