from abc import ABC
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Self, TypedDict

from numpy.typing import NDArray
from polars import DataFrame, Series
from polars._typing import IntoExpr
from sklearn import linear_model

from polars_ml.pipeline.component import PipelineComponent

from .utils import plot_feature_coefficients


class LinearRegressionParameters(TypedDict, total=False):
    fit_intercept: bool
    n_jobs: int
    positive: bool


class LinearRegressionFitArguments(TypedDict, total=False):
    sample_weight: NDArray[Any]


class LinearRegression(PipelineComponent, ABC):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        *,
        prediction_name: str = "linear_regression",
        include_input: bool = True,
        model_kwargs: LinearRegressionParameters
        | Callable[[DataFrame], LinearRegressionParameters]
        | None = None,
        fit_kwargs: LinearRegressionFitArguments
        | Callable[[DataFrame], LinearRegressionFitArguments]
        | None = None,
        out_dir: str | Path | None = None,
    ):
        self.features = features
        self.label = label
        self.prediction_name = prediction_name
        self.include_input = include_input
        self.model_kwargs = model_kwargs or {}
        self.fit_kwargs = fit_kwargs or {}
        self.out_dir = Path(out_dir) if out_dir is not None else None

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        train_features = data.select(self.features)
        train_label = data.select(self.label)
        self.feature_names = train_features.columns

        model_kwargs = (
            self.model_kwargs(data)
            if callable(self.model_kwargs)
            else self.model_kwargs
        )
        self.model = linear_model.LinearRegression(copy_X=False, **model_kwargs)

        X = train_features.to_numpy()
        y = train_label.to_numpy().squeeze()
        fit_kwargs = (
            self.fit_kwargs(data) if callable(self.fit_kwargs) else self.fit_kwargs
        )
        self.model.fit(X, y, **fit_kwargs)

        if self.out_dir is not None:
            self.save()
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input = data.select(self.features)
        pred = self.model.predict(input.to_numpy())

        if self.include_input:
            return data.with_columns(Series(self.prediction_name, pred))
        else:
            return DataFrame(Series(self.prediction_name, pred))

    def save(self, out_dir: str | Path | None = None):
        out_dir = Path(out_dir) if out_dir else self.out_dir
        if out_dir is None:
            raise ValueError("No output directory provided")

        out_dir.mkdir(parents=True, exist_ok=True)

        plot_feature_coefficients(
            self.model.coef_,
            self.feature_names,
            filepath=out_dir / "feature_coefficients.png",
        )

        coef_df = DataFrame(
            {"feature": self.feature_names, "coefficient": self.model.coef_}
        )
        coef_df.write_csv(out_dir / "feature_coefficients.csv")
