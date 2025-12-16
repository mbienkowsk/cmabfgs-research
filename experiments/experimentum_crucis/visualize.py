from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.funs import elliptic_hess_for_dim

DIM = 100

RESULT_PATH = (
    Path(__file__).parent / "results" / f"elliptic_d{DIM}" / f"elliptic_d{DIM}.parquet"
)


def get_eigv_ratio_statistics(arr: np.ndarray):
    ratios = np.divide(arr[1:], arr[:-1])
    return pd.Series(
        {
            "eigv_ratios": ratios,
            "eigv_ratios_mean": np.mean(ratios),
            "eigv_ratios_25th": np.percentile(ratios, 25),
            "eigv_ratios_75th": np.percentile(ratios, 75),
        }
    )


def visualize_results(df: pd.DataFrame):
    ratio_stats = df["cov_mat_eigv"].apply(get_eigv_ratio_statistics)
    with_ratio_stats = pd.concat((df, ratio_stats), axis=1)

    averaged = with_ratio_stats.drop(columns=["run_id"]).groupby(level=0).mean()
    ndim = len(averaged.iloc[0]["cov_mat_eigv"])

    plt.semilogy(averaged.index, averaged["best"])
    plt.title("Best so far vs fevals")
    plt.show()

    eigenvalue_array = np.array(list(averaged["cov_mat_eigv"]))

    for i in range(ndim):
        plt.semilogy(averaged.index, eigenvalue_array[:, i])
    plt.title("Raw eigenvalue plot")
    plt.show()

    for i in range(ndim):
        plt.semilogy(
            averaged.index, np.sqrt(eigenvalue_array[:, i]) * averaged["sigma"]
        )
    plt.title("sqrt(lambda) * sigma")
    plt.show()

    for i in range(ndim):
        plt.semilogy(
            averaged.index,
            np.sqrt(eigenvalue_array[:, ndim - 1] / eigenvalue_array[:, 0]),
        )
    plt.title("largest/smallest std dev ratio")
    plt.show()

    for i in range(ndim):
        plt.semilogy(averaged.index, eigenvalue_array[:, i] * averaged["sigma"] ** 2)

    plt.title("sigma^2 * lambda")
    plt.show()

    actual_eigenvalues, _ = np.linalg.eigh(elliptic_hess_for_dim(ndim))
    actual_ratios = np.divide(actual_eigenvalues[1:], actual_eigenvalues[:-1])
    actual_ratio_average = np.mean(actual_ratios)

    for i in range(len(actual_ratios)):
        plt.semilogy(
            averaged.index,
            np.full((len(averaged.index)), actual_ratios[i]),
            label=f"actual ratio {i + 1}",
            linestyle="dashed",
        )
        plt.semilogy(
            averaged["eigv_ratios"].apply(lambda x: x[i]),
            label=f"estimated ratio {i + 1}",
        )
    plt.title("eigenvalue ratios comparison")
    plt.legend()
    plt.show()

    plt.semilogy(averaged["eigv_ratios_mean"], label="mean")
    plt.semilogy(averaged["eigv_ratios_25th"], label="25th percentile")
    plt.semilogy(averaged["eigv_ratios_75th"], label="75th percentile")
    plt.semilogy(
        averaged.index,
        np.full((len(averaged.index)), actual_ratio_average),
        label="actual ratio mean",
        linestyle="dashed",
    )
    plt.title("eigenvalue ratios statistics")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    visualize_results(pd.read_parquet(RESULT_PATH))
