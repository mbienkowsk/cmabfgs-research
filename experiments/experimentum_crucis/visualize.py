from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.funs import elliptic_hess_for_dim


def get_plot_directory(dim: int):
    return Path(__file__).parent / "results" / "plots" / f"d_{dim}"


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


@contextmanager
def wrap_plot(title: str, dim: int, ylabel: str, save_to: Path):
    fig, ax = plt.subplots()
    secax = ax.secondary_xaxis(
        "bottom",
        functions=(lambda x: x / (4 * dim), lambda x: x * 4 * dim),  # pyright: ignore[reportOperatorIssue]
    )
    secax.set_xlabel("Secondary scale")
    secax.spines["bottom"].set_position(("outward", 40))
    yield
    plt.xlabel("liczba ewaluacji f.celu")
    secax.set_xlabel("Liczba iteracji algorytmu")
    plt.title(title + f" ({dim} wymiarów)")
    plt.grid()
    plt.savefig(save_to)


def visualize_results(df: pd.DataFrame, dim: int, save_dir: Path):
    ratio_stats = df["cov_mat_eigv"].apply(get_eigv_ratio_statistics)
    with_ratio_stats = pd.concat((df, ratio_stats), axis=1)

    averaged = with_ratio_stats.drop(columns=["run_id"]).groupby(level=0).mean()
    ndim = len(averaged.iloc[0]["cov_mat_eigv"])

    with wrap_plot(
        "Krzywa zbieżności algorytmu CMA-ES na f. pokrzywionej",
        dim,
        "f(xbest)",
        save_dir / "convergence_curve.png",
    ):
        plt.semilogy(averaged.index, averaged["best"])

    with wrap_plot(
        "Wartości własne macierzy kowariancji",
        dim,
        "$\\lambda_i$",
        save_dir / "raw_eigvals.png",
    ):
        eigenvalue_array = np.array(list(averaged["cov_mat_eigv"]))
        for i in range(ndim):
            plt.semilogy(averaged.index, eigenvalue_array[:, i])

    with wrap_plot(
        "długości półosi rozkładu populacji",
        dim,
        "sqrt(\\lambda_i) * \\sigma",
        save_dir / "axis_lengths.png",
    ):
        for i in range(ndim):
            plt.semilogy(
                averaged.index, np.sqrt(eigenvalue_array[:, i]) * averaged["sigma"]
            )

    with wrap_plot(
        "Iloraz $\\lambda_{max} / \\lambda_{min}$",
        dim,
        "stos. $\\lambda$",
        save_dir / "eig_max_min_ratio.png",
    ):
        for i in range(ndim):
            plt.semilogy(
                averaged.index,
                np.sqrt(eigenvalue_array[:, ndim - 1] / eigenvalue_array[:, 0]),
            )

    with wrap_plot(
        "$\\sigma^2 * \\lambda$",
        dim,
        "$\\sigma^2 * \\lambda$",
        save_dir / "axis_lengths_sq.png",
    ):
        for i in range(ndim):
            plt.semilogy(
                averaged.index, eigenvalue_array[:, i] * averaged["sigma"] ** 2
            )

    actual_eigenvalues, _ = np.linalg.eigh(elliptic_hess_for_dim(ndim))
    actual_ratios = np.divide(actual_eigenvalues[1:], actual_eigenvalues[:-1])
    actual_ratio_average = np.mean(actual_ratios)

    with wrap_plot(
        "Porównanie ilorazów kolejnych wartości własnych ($\\lambda[1:] / \\lambda[:-1]$)",
        dim,
        "$\\lambda[i] / \\ labmda[i-1]$",
        save_dir / "consecutive_ratios.png",
    ):
        for i in range(len(actual_ratios)):
            plt.semilogy(
                averaged.index,
                np.full((len(averaged.index)), actual_ratios[i]),
                label=f"actual ratio {i + 1}",
                linestyle="dashed",
            )
            plt.semilogy(
                averaged["eigv_ratios"].apply(lambda x: x[i]),  # pyright: ignore[reportAttributeAccessIssue]
                label=f"estimated ratio {i + 1}",
            )
        plt.legend()

    with wrap_plot(
        "Porównanie statystyk ilorazów kolejnych wartości własnych ($\\lambda[1:] / \\lambda[:-1]$)",
        dim,
        "$lambda[i] / \\ labmda[i-1]$",
        save_dir / "consecutive_ratios_stats.png",
    ):
        plt.semilogy(averaged["eigv_ratios_mean"], label="mean")
        plt.semilogy(averaged["eigv_ratios_25th"], label="25. centyl")
        plt.semilogy(averaged["eigv_ratios_75th"], label="75. centyl")
        plt.semilogy(
            averaged.index,
            np.full((len(averaged.index)), actual_ratio_average),
            label="actual ratio mean",
            linestyle="dashed",
        )
        plt.legend()

    averaged["corresp_eig_ratios"] = averaged["cov_mat_eigv"].apply(
        lambda eig: np.divide(eig, actual_eigenvalues)
    )

    with wrap_plot(
        "Porównanie ilorazów odpowiadających sobie wartości własnych C i H_inv",
        dim,
        "$\\lambda_{C}[i] / \\ lambda_{H_inv}[i]$",
        save_dir / "corresp_eig_ratios.png",
    ):
        for i in range(len(actual_eigenvalues)):
            plt.semilogy(
                averaged["corresp_eig_ratios"].apply(lambda x: x[i]),  # pyright: ignore[reportAttributeAccessIssue]
                label=f"ratio for $\\lambda_{i + 1}$",
            )
        plt.legend()

    with wrap_plot(
        "Zagregowane dane dotyczące ilorazów odpowiadających sobie wartości własnych C i H_inv",
        dim,
        "$\\lambda_{C}[i] / \\lambda_{H_inv}[i]$",
        save_dir / "corresp_eig_ratios_stats.png",
    ):
        ratios_df = pd.DataFrame(
            averaged["corresp_eig_ratios"].to_list(),
            index=averaged.index,
        )

        mean = ratios_df.mean(axis=1)
        q25 = ratios_df.quantile(0.25, axis=1)
        q75 = ratios_df.quantile(0.75, axis=1)

        plt.semilogy(mean.index, mean, label="mean")
        plt.semilogy(q25.index, q25, label="25. centyl")
        plt.semilogy(q75.index, q75, label="75. centyl")

        plt.legend()


if __name__ == "__main__":
    DIMS = [5, 10, 20, 50, 100]
    for dir in (get_plot_directory(dim) for dim in DIMS):
        dir.mkdir(parents=True, exist_ok=True)

    for dim in DIMS:
        RESULT_PATH = (
            Path(__file__).parent
            / "results"
            / f"elliptic_d{dim}"
            / f"elliptic_d{dim}.parquet"
        )
        df = pd.read_parquet(RESULT_PATH)
        df["iteration"] = df.index // 4 * dim
        visualize_results(pd.read_parquet(RESULT_PATH), dim, get_plot_directory(dim))
