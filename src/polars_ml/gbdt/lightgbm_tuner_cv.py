from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, Sequence

import numpy as np
import polars as pl
import polars.selectors as cs
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError

if TYPE_CHECKING:
    import lightgbm as lgb
    from optuna_integration.lightgbm import LightGBMTunerCV


class OptunaLightGBMTunerCV(Transformer):
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

        self._tuner_params: dict[str, Any] = {}
        self._dataset_params: dict[str, Any] = {}

        self._tuner: LightGBMTunerCV | None = None
        self._cvbooster: lgb.CVBooster | None = None

    def set_tuner_params(self, **kwargs: Any) -> Self:
        self._tuner_params = kwargs
        return self

    def set_dataset_params(self, **kwargs: Any) -> Self:
        self._dataset_params = kwargs
        return self

    def make_train_set(self, data: pl.DataFrame) -> lgb.Dataset:
        if self._features is None or self._target is None:
            raise NotFittedError()

        import lightgbm as lgb

        dtrain = lgb.Dataset(
            data.select(self._features).to_pandas(),
            label=data.select(self._target).to_pandas().squeeze(),
            feature_name=self._features,
            **self._dataset_params,
        )
        return dtrain

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        from optuna_integration.lightgbm import LightGBMTunerCV

        self._target = (
            data.lazy().select(self._target_selector).collect_schema().names()
        )
        self._features = (
            data.lazy().select(self._features_selector).collect_schema().names()
        )

        dtrain = self.make_train_set(data)

        self._tuner = LightGBMTunerCV(
            self._params, dtrain, return_cvbooster=True, **self._tuner_params
        )
        self._tuner.run()

        self._cvbooster = self._tuner.get_best_booster()

        if self._fit_dir:
            self._fit_dir.mkdir(parents=True, exist_ok=True)
            for i, booster in enumerate(self._cvbooster.boosters):
                booster.save_model(str(self._fit_dir / f"model_fold{i}.txt"))

        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        if self._features is None or self._cvbooster is None:
            raise NotFittedError()

        input_data = data.select(self._features).to_pandas()

        preds_list = self._cvbooster.predict(input_data)  # type: ignore

        ensemble_pred = np.mean(preds_list, axis=0)

        return pl.from_numpy(ensemble_pred, schema=self._prediction)

    @property
    def best_params(self) -> dict[str, Any]:
        if self._tuner is None:
            raise NotFittedError()
        return self._tuner.best_params

    @property
    def best_score(self) -> float:
        if self._tuner is None:
            raise NotFittedError()
        return self._tuner.best_score
