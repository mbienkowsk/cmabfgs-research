from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DIM = 10
RESULT_PATH = Path(__file__).parent / "results" / f"d{DIM}" / "agg"


def visualize_results(df: pd.DataFrame):
    df.plot(
        title="BFGS inicjowany macierzą kowariancji CMAESa po n iteracjach - porównanie krzywych zbieżności"
    )
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    visualize_results(pd.read_parquet(RESULT_PATH / "bfgs.parquet"))
    visualize_results(pd.read_parquet(RESULT_PATH / "bfgs_inherited_x0.parquet"))
