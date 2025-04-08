from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Self

import numpy as np
from polars import DataFrame, Series
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    from sklearn import decomposition


class PCA(PipelineComponent):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        pca: "decomposition.PCA",
        *,
        prefix: str = "pca",
        include_input: bool = True,
        out_dir: str | Path | None = None,
    ):
        self.features = features
        self.pca = pca
        self.prefix = prefix
        self.include_input = include_input
        self.out_dir = Path(out_dir) if out_dir is not None else None

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        train_features = data.select(self.features)
        self.feature_names = train_features.columns

        X = train_features.to_numpy()
        self.pca.fit(X)

        if self.out_dir is not None:
            self.save()

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input_data = data.select(self.features)
        transformed = self.pca.transform(input_data.to_numpy())

        pc_columns = [
            Series(f"{self.prefix}_{i}", transformed[:, i])
            for i in range(transformed.shape[1])
        ]

        if self.include_input:
            return data.with_columns(pc_columns)
        else:
            return DataFrame(pc_columns)

    def save(self, out_dir: str | Path | None = None):
        out_dir = Path(out_dir) if out_dir else self.out_dir
        if out_dir is None:
            raise ValueError("No output directory provided")

        out_dir.mkdir(parents=True, exist_ok=True)

        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        variance_df = DataFrame(
            {
                "component": [
                    f"{self.prefix}_{i + 1}"
                    for i in range(len(explained_variance_ratio))
                ],
                "explained_variance_ratio": explained_variance_ratio,
                "cumulative_variance_ratio": cumulative_variance_ratio,
            }
        )
        variance_df.write_csv(out_dir / "explained_variance.csv")

        components_df = DataFrame(
            {
                "feature": self.feature_names,
                **{
                    f"{self.prefix}_{i}": self.pca.components_[i]
                    for i in range(self.pca.components_.shape[0])
                },
            }
        )
        components_df.write_csv(out_dir / "components.csv")

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        plt.plot(
            range(1, len(cumulative_variance_ratio) + 1),
            cumulative_variance_ratio,
            "ro-",
        )
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Scree Plot")
        plt.grid(True)
        plt.savefig(out_dir / "scree_plot.png")
        plt.close()

        if self.pca.components_.shape[0] >= 2:
            plt.figure(figsize=(12, 10))

            plt.scatter(self.pca.components_[0], self.pca.components_[1])

            for i, txt in enumerate(self.feature_names):
                plt.annotate(
                    txt, (self.pca.components_[0, i], self.pca.components_[1, i])
                )

            plt.xlabel(f"{self.prefix}_1")
            plt.ylabel(f"{self.prefix}_2")
            plt.title("Feature Loadings (PC1 vs PC2)")
            plt.grid(True)
            plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
            plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
            plt.savefig(out_dir / "feature_loadings.png")
            plt.close()
