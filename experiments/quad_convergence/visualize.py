from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DIM = 10
RESULT_PATH = Path(__file__).parent / "results" / f"d{DIM}" / "agg"


def visualize_results(df: pd.DataFrame):
    normalized_cols = [col for col in df.columns if "normalized" in col]
    df.loc[:400].plot(
        y=[col for col in df.columns if col not in normalized_cols],
        title="BFGS inicjowany macierzą kowariancji CMAESa po n iteracjach - porównanie krzywych zbieżności",
        logy=True,
    )
    df.loc[:400].plot(
        y=normalized_cols,
        title="BFGS inicjowany znormalizowaną macierzą kowariancji CMAESa po n iteracjach - porównanie krzywych zbieżności",
        logy=True,
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    visualize_results(pd.read_parquet(RESULT_PATH / "bfgs.parquet"))
    visualize_results(pd.read_parquet(RESULT_PATH / "bfgs_inherited_x0.parquet"))
