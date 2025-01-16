from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Self, Type

import polars as pl
from numpy.typing import NDArray
from polars import DataFrame, Series
from polars._typing import IntoExpr

from polars_ml import Component

if TYPE_CHECKING:
    from sklearn import decomposition


class DecompositionModel(Component, ABC):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        components_name: str,
        append_components: bool,
        model_class: Type[Any],
        model_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        self.features = features
        self.components_name = components_name
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
            (f"{self.components_name}_{i}", Series(components[:, i]))
            for i in range(components.shape[1])
        ]

        if self.append_components:
            return data.with_columns(new_columns)
        else:
            return DataFrame(new_columns)


class PCA(DecompositionModel):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        components_name: str = "pca",
        append_components: bool = True,
        model_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        super().__init__(
            features=features,
            components_name=components_name,
            append_components=append_components,
            model_class=decomposition.PCA,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )


class NMF(DecompositionModel):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        components_name: str = "nmf",
        append_components: bool = True,
        model_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        super().__init__(
            features=features,
            components_name=components_name,
            append_components=append_components,
            model_class=decomposition.NMF,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )


class TruncatedSVD(DecompositionModel):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        components_name: str = "truncated_svd",
        append_components: bool = True,
        model_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        super().__init__(
            features=features,
            components_name=components_name,
            append_components=append_components,
            model_class=decomposition.TruncatedSVD,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )
