from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self

from numpy.typing import NDArray
from polars import DataFrame, Series
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    import umap


class UMAP(PipelineComponent):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        umap: "umap.UMAP",
        *,
        prefix: str = "umap",
        include_input: bool = True,
        out_dir: str | Path | None = None,
    ):
        self.features = features
        self.umap = umap
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
        self.umap.fit(X)

        if self.out_dir is not None:
            self.save()

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input_data = data.select(self.features)
        transformed: NDArray[Any] = self.umap.transform(input_data.to_numpy())  # type: ignore

        umap_columns = [
            Series(f"{self.prefix}_{i}", transformed[:, i])
            for i in range(transformed.shape[1])
        ]

        if self.include_input:
            return data.with_columns(umap_columns)
        else:
            return DataFrame(umap_columns)

    def save(self, out_dir: str | Path | None = None):
        import joblib
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir) if out_dir else self.out_dir
        if out_dir is None:
            raise ValueError("No output directory provided")

        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.umap, out_dir / "umap.pkl")

        if self.umap.embedding_.shape[1] >= 2:
            plt.figure(figsize=(12, 10))
            plt.scatter(self.umap.embedding_[:, 0], self.umap.embedding_[:, 1], s=5)
            plt.title("UMAP Embedding")
            plt.xlabel(f"{self.prefix}_1")
            plt.ylabel(f"{self.prefix}_2")
            plt.grid(True)
            plt.savefig(out_dir / "umap_embedding.png")
            plt.close()
