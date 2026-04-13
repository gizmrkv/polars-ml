from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, Sequence

import polars as pl
import polars.selectors as cs
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError

if TYPE_CHECKING:
    import xgboost as xgb


class XGBoost(Transformer):
    def __init__(
        self,
        target: ColumnNameOrSelector,
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        prediction: str | Sequence[str] | None = None,
        *,
        params: Mapping[str, Any],
        fit_dir: str | Path | None = None,
    ):
        self._target_selector = target
        self._prediction = (
            [prediction]
            if isinstance(prediction, str)
            else ["prediction"]
            if prediction is None
            else list(prediction)
        )
        self._features_selector = (
            features if features is not None else cs.exclude(target)
        )
        self._params = dict(params)
        self._fit_dir = Path(fit_dir) if fit_dir else None

        self._target: list[str] | None = None
        self._features: list[str] | None = None
        self._train_params: dict[str, Any] = {}
        self._dmatrix_params: dict[str, Any] = {}
        self._booster: xgb.Booster | None = None

    def set_train_params(self, **kwargs: Any) -> Self:
        self._train_params = kwargs
        return self

    def set_dmatrix_params(self, **kwargs: Any) -> Self:
        self._dmatrix_params = kwargs
        return self

    def make_train_valid_sets(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> tuple[xgb.DMatrix, list[tuple[xgb.DMatrix, str]]]:
        if self._features is None or self._target is None:
            raise NotFittedError()

        import xgboost as xgb

        dtrain = xgb.DMatrix(
            data.select(*self._features).to_pandas(),
            label=data.select(*self._target).to_pandas(),
            feature_names=self._features,
            **self._dmatrix_params,
        )
        evals = []
        for name, valid_data in more_data.items():
            dvalid = xgb.DMatrix(
                valid_data.select(*self._features).to_pandas(),
                label=valid_data.select(*self._target).to_pandas(),
                feature_names=self._features,
                **self._dmatrix_params,
            )
            evals.append((dvalid, name))

        return dtrain, evals

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        import xgboost as xgb

        self._target = (
            data.lazy().select(self._target_selector).collect_schema().names()
        )
        self._features = (
            data.lazy().select(self._features_selector).collect_schema().names()
        )

        dtrain, evals = self.make_train_valid_sets(data, **more_data)
        self._booster = xgb.train(
            self._params, dtrain, evals=evals, **self._train_params
        )

        if self._fit_dir:
            self._booster.save_model(self._fit_dir / "model.txt")

        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        if self._features is None or self._booster is None:
            raise NotFittedError()

        import xgboost as xgb

        input_data = xgb.DMatrix(
            data.select(self._features).to_pandas(),
            feature_names=self._features,
        )
        pred = self._booster.predict(input_data)
        return pl.from_numpy(pred, schema=self._prediction)
