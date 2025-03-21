from pathlib import Path
from typing import Any

from numpy.typing import NDArray


def plot_feature_coefficients(
    coefficients: list[float] | NDArray[Any],
    feature_names: list[str],
    *,
    filepath: str | Path,
    figsize: tuple[float, float] | None = None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    coefficients = np.array(coefficients)

    if len(coefficients) != len(feature_names):
        raise ValueError(
            f"Length of coefficients ({len(coefficients)}) and feature names ({len(feature_names)}) must be equal"
        )

    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
    sorted_coefficients = coefficients[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]

    if figsize is None:
        figsize = (10, max(8, len(feature_names) * 0.3))

    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(sorted_coefficients)), sorted_coefficients)

    for i, bar in enumerate(bars):
        if sorted_coefficients[i] < 0:
            bar.set_color("r")
        else:
            bar.set_color("b")

    plt.yticks(range(len(sorted_coefficients)), sorted_feature_names)
    plt.xlabel("Coefficient Value")
    plt.title("Feature Coefficients (sorted by absolute value)")
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
