from abc import ABC
from typing import Any, Callable, Iterable, Mapping, Self, Type

from numpy.typing import NDArray
from polars import DataFrame, Series
from polars._typing import IntoExpr

from polars_ml import Component


class LinearModel(Component, ABC):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        *,
        prediction_name: str,
        append_prediction: bool,
        model_class: Type[Any],
        model_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        self.features = features
        self.label = label
        self.prediction_name = prediction_name
        self.append_prediction = append_prediction
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.fit_kwargs = fit_kwargs or {}

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        train_features = data.select(self.features)
        train_label = data.select(self.label)

        model_kwargs = (
            self.model_kwargs(data)
            if callable(self.model_kwargs)
            else self.model_kwargs
        )
        self.model = self.model_class(**model_kwargs)

        fit_kwargs = (
            self.fit_kwargs(data) if callable(self.fit_kwargs) else self.fit_kwargs
        )
        self.model.fit(
            train_features.to_numpy(), train_label.to_numpy().squeeze(), **fit_kwargs
        )
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input = data.select(self.features)
        pred: NDArray[Any] = self.model.predict(input.to_numpy())

        if self.append_prediction:
            return data.with_columns(Series(self.prediction_name, pred))
        else:
            return DataFrame(Series(self.prediction_name, pred))


class LinearRegression(LinearModel):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        prediction_name: str = "linear_regression",
        append_prediction: bool = True,
        model_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        from sklearn import linear_model

        super().__init__(
            features,
            label,
            prediction_name=prediction_name,
            append_prediction=append_prediction,
            model_class=linear_model.LinearRegression,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )


class Ridge(LinearModel):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        prediction_name: str = "ridge",
        append_prediction: bool = True,
        model_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        from sklearn import linear_model

        super().__init__(
            features,
            label,
            prediction_name=prediction_name,
            append_prediction=append_prediction,
            model_class=linear_model.Ridge,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )


class Lasso(LinearModel):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        prediction_name: str = "lasso",
        append_prediction: bool = True,
        model_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        from sklearn import linear_model

        super().__init__(
            features,
            label,
            prediction_name=prediction_name,
            append_prediction=append_prediction,
            model_class=linear_model.Lasso,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )


class ElasticNet(LinearModel):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        prediction_name: str = "elastic_net",
        append_prediction: bool = True,
        model_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ):
        from sklearn import linear_model

        super().__init__(
            features,
            label,
            prediction_name=prediction_name,
            append_prediction=append_prediction,
            model_class=linear_model.ElasticNet,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )
