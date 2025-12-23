from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DIM = 10

RESULT_PATH = Path(__file__).parent / "results" / f"d{DIM}" / f"bfgs_d{DIM}_raw.parquet"


def visualize_results(df: pd.DataFrame):
    df[df.index < 400].plot(
        title="BFGS inicjowany macierzą kowariancji CMAESa po n iteracjach - porównanie krzywych zbieżności"
    )
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    visualize_results(pd.read_parquet(RESULT_PATH))
