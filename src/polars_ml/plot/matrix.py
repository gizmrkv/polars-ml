import itertools
import uuid
from typing import Any, Dict, Tuple, override

import numpy as np
import polars as pl
import polars.selectors as cs
from polars import DataFrame

from polars_ml.component import Component


class CorrelationMatrix(Component):
    def __init__(
        self,
        *,
        figsize: Tuple[int, int] = (22, 20),
        heatmap_kws: Dict[str, Any] | None = None,
    ):
        self.figsize = figsize
        self.heatmap_kws = heatmap_kws or {
            "cmap": "coolwarm",
            "vmax": 1.0,
            "vmin": -1.0,
            "center": 0,
            "square": True,
            "linewidths": 0.5,
            "xticklabels": True,
            "yticklabels": True,
            "cbar_kws": {"shrink": 0.5},
        }
        self._is_fitted = True

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        if log_dir := self.log_dir:
            import matplotlib.pyplot as plt
            import seaborn as sns

            corr = data.select(cs.numeric()).corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))

            plt.figure(figsize=self.figsize)
            sns.heatmap(corr, mask=mask, **self.heatmap_kws)

            plt.xticks(
                ticks=np.arange(len(corr.columns)) + 0.5,
                labels=corr.columns,
                rotation=90,
            )
            plt.yticks(
                ticks=np.arange(len(corr.columns)) + 0.5,
                labels=corr.columns,
                rotation=0,
            )

            plt.title("Correlation Matrix")
            plt.tight_layout()

            log_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(log_dir / "correlation_matrix.png")
            plt.clf()
            plt.close()

        return data


class NullMatrix(Component):
    def __init__(
        self,
        *,
        figsize: Tuple[int, int] = (20, 20),
        heatmap_kws: Dict[str, Any] | None = None,
    ):
        self.figsize = figsize
        self.heatmap_kws = heatmap_kws or {"cbar": False}

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        if log_dir := self.log_dir:
            import matplotlib.pyplot as plt
            import seaborn as sns

            null = data.select(pl.all().is_null())
            plt.figure(figsize=self.figsize)
            sns.heatmap(null, **self.heatmap_kws)

            plt.xticks(
                ticks=np.arange(len(null.columns)) + 0.5,
                labels=null.columns,
                rotation=90,
            )
            plt.yticks(
                ticks=np.arange(len(null.columns)) + 0.5,
                labels=null.columns,
                rotation=0,
            )

            plt.tight_layout()

            log_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(log_dir / "null_matrix.png")
            plt.clf()
            plt.close()

        return data


class CategoricalIndependencyMatrix(Component):
    def __init__(
        self,
        *,
        figsize: Tuple[int, int] = (22, 20),
        heatmap_kws: Dict[str, Any] | None = None,
    ):
        self.figsize = figsize
        self.heatmap_kws = heatmap_kws or {
            "cmap": "viridis",
            "vmax": 1.0,
            "vmin": 0.0,
            "center": 0,
            "square": True,
            "linewidths": 0.5,
            "xticklabels": True,
            "yticklabels": True,
            "cbar_kws": {"shrink": 0.5},
        }

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        if log_dir := self.log_dir:
            import matplotlib.pyplot as plt
            import seaborn as sns

            n_uniques = data.select(pl.all().n_unique()).row(0, named=True)
            n_uniques_pair: Dict[str, Dict[str, float]] = {}
            pair_index = uuid.uuid4().hex
            for c1, c2 in itertools.combinations_with_replacement(data.columns, 2):
                n_unique_min = max(n_uniques[c1], n_uniques[c2])
                n_unique_max = min(n_uniques[c1] * n_uniques[c2], data.height)
                independency = (
                    data.select(
                        (pl.col(c1) if c1 == c2 else pl.struct(c1, c2)).alias(
                            pair_index
                        )
                    )[pair_index].n_unique()
                    - n_unique_min
                    - 1
                ) / max(n_unique_max - n_unique_min, 1)

                n_uniques_pair.setdefault(c1, {})[c2] = independency
                n_uniques_pair.setdefault(c2, {})[c1] = independency

            matrix = pl.DataFrame(
                {k1: [v2 for v2 in v1.values()] for k1, v1 in n_uniques_pair.items()}
            )
            columns = matrix.columns
            mask = np.triu(np.ones_like(matrix, dtype=bool))

            plt.figure(figsize=self.figsize)
            sns.heatmap(matrix, mask=mask, **self.heatmap_kws)
            plt.xticks(labels=columns, rotation=90)
            plt.yticks(labels=columns, rotation=0)
            plt.title("Categorical Independency Matrix")
            plt.tight_layout()

            log_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(log_dir / "categorical_independency_matrix.png")
            plt.clf()
            plt.close()

        return data
