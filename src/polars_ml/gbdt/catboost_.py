from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, Sequence

import polars as pl
import polars.selectors as cs
from polars import DataFrame
from polars._typing import ColumnNameOrSelector

from polars_ml.base import HasFeatureImportance, Transformer
from polars_ml.exceptions import NotFittedError

if TYPE_CHECKING:
    import catboost as cb


class CatBoost(Transformer, HasFeatureImportance):
    def __init__(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        params: Mapping[str, Any] | None = None,
        fit_dir: str | Path | None = None,
        **fit_params: Any,
    ):
        self._target_selector = target
        self._prediction = (
            [prediction] if isinstance(prediction, str) else list(prediction)
        )
        self._features_selector = (
            features if features is not None else cs.exclude(target)
        )
        self._params = dict(params) if params else {}
        self._fit_dir = Path(fit_dir) if fit_dir else None
        self._fit_params = fit_params

        self._target: list[str] | None = None
        self._features: list[str] | None = None
        self._pool_params: dict[str, Any] | None = None
        self._booster: cb.CatBoost | None = None

    @property
    def target(self) -> list[str]:
        if self._target is None:
            raise NotFittedError()
        return self._target

    @property
    def features(self) -> list[str]:
        if self._features is None:
            raise NotFittedError()
        return self._features

    @property
    def pool_params(self) -> dict[str, Any]:
        if self._pool_params is None:
            raise NotFittedError()
        return self._pool_params

    @property
    def booster(self) -> cb.CatBoost:
        if self._booster is None:
            raise NotFittedError()
        return self._booster

    def init_target(self, data: DataFrame) -> list[str]:
        return data.lazy().select(self._target_selector).collect_schema().names()

    def init_features(self, data: DataFrame) -> list[str]:
        return data.lazy().select(self._features_selector).collect_schema().names()

    def init_fit_params(self, data: DataFrame) -> dict[str, Any]:
        return self._fit_params

    def init_pool_params(self, data: DataFrame) -> dict[str, Any]:
        return {}

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        import catboost as cb

        self._target = self.init_target(data)
        self._features = self.init_features(data)
        self._pool_params = self.init_pool_params(data)
        self._fit_params = self.init_fit_params(data)

        train_pool = cb.Pool(
            data=data.select(*self.features).to_pandas(),
            label=data.select(*self.target).to_pandas(),
            feature_names=self._features,
            **self.pool_params,
        )
        eval_sets = []
        for valid_data in more_data.values():
            eval_sets.append(
                cb.Pool(
                    data=valid_data.select(*self.features).to_pandas(),
                    label=valid_data.select(*self.target).to_pandas(),
                    feature_names=self._features,
                    **self.pool_params,
                )
            )

        self._booster = cb.CatBoost(self._params)
        self.booster.fit(
            train_pool,
            eval_set=eval_sets if eval_sets else None,
            **self._fit_params,
        )

        if self._fit_dir:
            self.save(self._fit_dir)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input_data = data.select(*self.features).to_pandas()
        pred = self.booster.predict(input_data)
        return pl.from_numpy(pred, schema=self._prediction)

    def save(self, fit_dir: str | Path):
        fit_dir = Path(fit_dir)
        fit_dir.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(fit_dir / "model.cbm"))

    def get_feature_importance(self) -> DataFrame:
        importance = self.booster.get_feature_importance()
        return DataFrame(
            {
                "feature": self.features,
                "importance": importance,
            }
        )
