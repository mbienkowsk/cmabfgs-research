from scipy.optimize import OptimizeResult, minimize

from lib.optimizers.bfgs import BFGS


class LBFGS(BFGS):

    def optimize(self):

        def callback_wrapper(intermediate_result: OptimizeResult):
            self.state.current_result = intermediate_result
            return self.callback(self.state)

        minimize(
            self.state.counter,
            self.x0,
            method="L-BFGS-B",
            callback=callback_wrapper,
        )
