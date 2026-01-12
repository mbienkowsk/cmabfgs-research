import matplotlib.pyplot as plt
import numpy as np

import lib.metrics as m
from lib.funs import get_function_by_name
from lib.metrics_collector import MetricsCollector
from lib.optimizers.cmaes import CMAES
from lib.random import IndividualGenerator
from lib.stopping import CMAESEarlyStopping
from lib.util import EvalCounter, evaluation_budget

DIMENSIONS = 2
POPSIZE = 4 * DIMENSIONS
BOUNDS = (-100.0, 100.0)
VISUALIZE_AT = [1, 10, 20, 40]


def visualize_population(
    population: list[np.ndarray],
    generation: int,
    objective,
) -> None:
    population = np.array(population)
    plt.figure(figsize=(6, 6))

    plot_objective_contours(objective, BOUNDS)

    plt.scatter(
        population[:, 0],
        population[:, 1],
        c="red",
        label="Population",
        zorder=3,
    )
    plt.xlim(BOUNDS)
    plt.ylim(BOUNDS)
    plt.title(f"CMA-ES Population at Generation {generation}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()


def plot_objective_contours(
    objective,
    bounds,
    resolution: int = 200,
) -> None:
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [
            [objective(np.array([xx, yy])) for xx, yy in zip(row_x, row_y)]
            for row_x, row_y in zip(X, Y)
        ]
    )
    plt.contour(X, Y, Z, levels=30, cmap="jet")
    plt.colorbar(label="Objective Value")


if __name__ == "__main__":
    RUN_ID = 1
    generator = IndividualGenerator(RUN_ID, BOUNDS, DIMENSIONS)
    objective = EvalCounter(get_function_by_name("Elliptic", DIMENSIONS))  # pyright: ignore[reportArgumentType]
    metrics = [m.CMAESIteration(POPSIZE), m.CMAESPopulation()]
    callback = MetricsCollector(metrics, 1)
    cmaes = CMAES(
        objective,
        generator.get_individual(),
        POPSIZE,
        generator.seed,
        CMAESEarlyStopping(max_evals=evaluation_budget(DIMENSIONS)),
        [callback],
        BOUNDS,
    )
    cmaes.optimize()
    data = callback.as_dataframe()

    for iters in VISUALIZE_AT:
        population = data.loc[data["iteration"] == iters, "population"].values[0]
        visualize_population(population, iters, objective)
