from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, Sequence

import polars as pl
import polars.selectors as cs
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError

if TYPE_CHECKING:
    import catboost as cb


class CatBoost(Transformer):
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
        self._pool_params: dict[str, Any] = {}
        self._model: cb.CatBoost | None = None

    def set_train_params(self, **kwargs: Any) -> Self:
        self._train_params = kwargs
        return self

    def set_pool_params(self, **kwargs: Any) -> Self:
        self._pool_params = kwargs
        return self

    def make_train_valid_sets(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> tuple[cb.Pool, list[cb.Pool]]:
        if self._features is None or self._target is None:
            raise NotFittedError()

        import catboost as cb

        dtrain = cb.Pool(
            data.select(self._features).to_pandas(),
            label=data.select(self._target).to_pandas(),
            feature_names=self._features,
            **self._pool_params,
        )

        evals = []
        for _, valid_data in more_data.items():
            dvalid = cb.Pool(
                valid_data.select(self._features).to_pandas(),
                label=valid_data.select(self._target).to_pandas(),
                feature_names=self._features,
                **self._pool_params,
            )
            evals.append(dvalid)

        return dtrain, evals

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        import catboost as cb

        self._target = (
            data.lazy().select(self._target_selector).collect_schema().names()
        )
        self._features = (
            data.lazy().select(self._features_selector).collect_schema().names()
        )

        dtrain, evals = self.make_train_valid_sets(data, **more_data)

        self._model = cb.CatBoost(self._params)
        self._model.fit(dtrain, eval_set=evals if evals else None, **self._train_params)

        if self._fit_dir:
            self._fit_dir.mkdir(parents=True, exist_ok=True)
            self._model.save_model(str(self._fit_dir / "model.cbm"))

        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        if self._features is None or self._model is None:
            raise NotFittedError()

        input_data = data.select(self._features).to_pandas()
        pred = self._model.predict(input_data)

        return pl.from_numpy(pred, schema=self._prediction)
