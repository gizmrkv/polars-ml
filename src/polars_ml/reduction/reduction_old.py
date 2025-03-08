from abc import ABC
from typing import Any, Callable, Iterable, Mapping, Self, Type

from numpy.typing import NDArray
from polars import DataFrame, Series
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent


class ReductionModel(PipelineComponent, ABC):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        output_name: str,
        append_components: bool,
        model_class: Type[Any],
        model_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        self.features = features
        self.output_name = output_name
        self.append_components = append_components
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.fit_kwargs = fit_kwargs or {}

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        train_data = data.select(self.features)

        model_kwargs = (
            self.model_kwargs(data)
            if callable(self.model_kwargs)
            else self.model_kwargs
        )
        self.model = self.model_class(**model_kwargs)

        fit_kwargs = (
            self.fit_kwargs(data) if callable(self.fit_kwargs) else self.fit_kwargs
        )
        self.model.fit(train_data.to_numpy(), **fit_kwargs)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input = data.select(self.features)
        components: NDArray[Any] = self.model.transform(input.to_numpy())

        new_columns = [
            Series(f"{self.output_name}_{i}", components[:, i])
            for i in range(components.shape[1])
        ]

        if self.append_components:
            return data.with_columns(new_columns)
        else:
            return DataFrame(new_columns)


class PCA(ReductionModel):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        output_name: str = "pca",
        append_components: bool = True,
        model_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        from sklearn import decomposition

        super().__init__(
            features=features,
            output_name=output_name,
            append_components=append_components,
            model_class=decomposition.PCA,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )


class NMF(ReductionModel):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        output_name: str = "nmf",
        append_components: bool = True,
        model_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        from sklearn import decomposition

        super().__init__(
            features=features,
            output_name=output_name,
            append_components=append_components,
            model_class=decomposition.NMF,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )


class TruncatedSVD(ReductionModel):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        output_name: str = "truncated_svd",
        append_components: bool = True,
        model_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        from sklearn import decomposition

        super().__init__(
            features=features,
            output_name=output_name,
            append_components=append_components,
            model_class=decomposition.TruncatedSVD,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )


class UMAP(ReductionModel):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        output_name: str = "umap",
        append_components: bool = True,
        model_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        import umap

        super().__init__(
            features=features,
            output_name=output_name,
            append_components=append_components,
            model_class=umap.UMAP,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )
