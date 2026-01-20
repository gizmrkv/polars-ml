from __future__ import annotations

from typing import Any, Iterable, Sequence

from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import IntoExpr
from sklearn import linear_model

from .base import BaseLinear


class LogisticRegression(BaseLinear):
    def __init__(
        self,
        model: linear_model.LogisticRegression | linear_model.LogisticRegressionCV,
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        prediction_name: str | Sequence[str] = "prediction",
    ):
        super().__init__(
            model,
            label,
            features,
            prediction_name=prediction_name,
        )

    def get_model(
        self,
    ) -> linear_model.LogisticRegression | linear_model.LogisticRegressionCV:
        return self.model

    def predict(self, data: DataFrame) -> NDArray[Any]:
        input_data = data.select(self.feature_names).to_pandas()
        return self.get_model().predict_proba(input_data)
