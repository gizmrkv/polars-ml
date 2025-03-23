from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Self, TypedDict

import umap
from numpy.typing import NDArray
from polars import DataFrame, Series
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent

UMAPMetricOptions = Literal[
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "canberra",
    "braycurtis",
    "mahalanobis",
    "wminkowski",
    "seuclidean",
    "cosine",
    "correlation",
    "haversine",
    "hamming",
    "jaccard",
    "dice",
    "russelrao",
    "kulsinski",
    "ll_dirichlet",
    "hellinger",
    "rogerstanimoto",
    "sokalmichener",
    "sokalsneath",
    "yule",
]


class UMAPParameters(TypedDict, total=False):
    n_neighbors: int
    n_components: int
    metric: UMAPMetricOptions
    n_epochs: int
    learning_rate: float
    init: Literal["spectral", "random", "pca", "tswspectral"]
    min_dist: float
    spread: float
    low_memory: bool
    set_op_mix_ratio: float
    local_connectivity: float
    repulsion_strength: float
    negative_sample_rate: int
    transform_queue_size: float
    a: float | None
    b: float | None
    random_state: int | None
    metric_kwds: dict[str, Any] | None
    angular_rp_forest: bool
    target_n_neighbors: int
    target_metric: str
    target_metric_kwds: dict[str, Any] | None
    target_weight: float
    transform_seed: int
    verbose: bool
    tqdm_kwds: dict[str, Any] | None
    unique: bool
    densmap: bool
    dens_lambda: float
    dens_frac: float
    dens_var_shift: float
    output_dens: bool
    disconnection_distance: float
    precomputed_knn: tuple[Any, ...]


class UMAP(PipelineComponent):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        prefix: str = "umap",
        include_input: bool = True,
        model_kwargs: UMAPParameters
        | Callable[[DataFrame], UMAPParameters]
        | None = None,
        out_dir: str | Path | None = None,
    ):
        self.features = features
        self.prefix = prefix
        self.include_input = include_input
        self.model_kwargs = model_kwargs or {}
        self.out_dir = Path(out_dir) if out_dir is not None else None

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        train_features = data.select(self.features)
        self.feature_names = train_features.columns

        model_kwargs = (
            self.model_kwargs(data)
            if callable(self.model_kwargs)
            else self.model_kwargs
        )
        self.model = umap.UMAP(**model_kwargs)

        X = train_features.to_numpy()
        self.model.fit(X)

        if self.out_dir is not None:
            self.save()

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input_data = data.select(self.features)
        transformed: NDArray[Any] = self.model.transform(input_data.to_numpy())  # type: ignore

        umap_columns = [
            Series(f"{self.prefix}_{i}", transformed[:, i])
            for i in range(transformed.shape[1])
        ]

        if self.include_input:
            return data.with_columns(umap_columns)
        else:
            return DataFrame(umap_columns)

    def save(self, out_dir: str | Path | None = None):
        out_dir = Path(out_dir) if out_dir else self.out_dir
        if out_dir is None:
            raise ValueError("No output directory provided")

        out_dir.mkdir(parents=True, exist_ok=True)

        import joblib

        joblib.dump(self.model, out_dir / "umap.pkl")

        import matplotlib.pyplot as plt

        if self.model.embedding_.shape[1] >= 2:
            plt.figure(figsize=(12, 10))
            plt.scatter(self.model.embedding_[:, 0], self.model.embedding_[:, 1], s=5)
            plt.title("UMAP Embedding")
            plt.xlabel(f"{self.prefix}_1")
            plt.ylabel(f"{self.prefix}_2")
            plt.grid(True)
            plt.savefig(out_dir / "umap_embedding.png")
            plt.close()
