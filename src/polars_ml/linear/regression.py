from __future__ import annotations

from typing import Iterable, Sequence

from polars._typing import IntoExpr
from sklearn import linear_model

from .base import BaseLinear

ScikitLearnLinearModel = (
    linear_model.LinearRegression
    | linear_model.Ridge
    | linear_model.RidgeCV
    | linear_model.Lasso
    | linear_model.LassoCV
    | linear_model.ElasticNet
    | linear_model.ElasticNetCV
    | linear_model.SGDRegressor
)


class LinearRegression(BaseLinear):
    def __init__(
        self,
        model: ScikitLearnLinearModel,
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        prediction_name: str | Sequence[str] = "prediction",
    ) -> None:
        super().__init__(
            model,
            label,
            features,
            prediction_name=prediction_name,
        )

    def get_model(self) -> ScikitLearnLinearModel:
        return self.model
