from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, Sequence

import polars as pl
import polars.selectors as cs
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError

if TYPE_CHECKING:
    import lightgbm as lgb


class LightGBM(Transformer):
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
        self._dataset_params: dict[str, Any] = {}
        self._booster: lgb.Booster | None = None

    def set_train_params(self, **kwargs: Any) -> Self:
        self._train_params = kwargs
        return self

    def set_dataset_params(self, **kwargs: Any) -> Self:
        self._dataset_params = kwargs
        return self

    def make_train_valid_sets(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> tuple[lgb.Dataset, list[lgb.Dataset], list[str]]:
        if self._features is None or self._target is None:
            raise NotFittedError()

        import lightgbm as lgb

        dtrain = lgb.Dataset(
            data.select(self._features).to_pandas(),
            label=data.select(self._target).to_pandas().squeeze(),
            feature_name=self._features,
            **self._dataset_params,
        )
        valid_sets = []
        valid_names = []
        for name, valid_data in more_data.items():
            dvalid = lgb.Dataset(
                valid_data.select(self._features).to_pandas(),
                label=valid_data.select(self._target).to_pandas().squeeze(),
                reference=dtrain,
                feature_name=self._features,
                **self._dataset_params,
            )
            valid_sets.append(dvalid)
            valid_names.append(name)

        return dtrain, valid_sets, valid_names

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        import lightgbm as lgb

        self._target = (
            data.lazy().select(self._target_selector).collect_schema().names()
        )
        self._features = (
            data.lazy().select(self._features_selector).collect_schema().names()
        )

        dtrain, valid_sets, valid_names = self.make_train_valid_sets(data, **more_data)

        self._booster = lgb.train(
            params=self._params,
            train_set=dtrain,
            valid_sets=valid_sets if valid_sets else None,
            valid_names=valid_names if valid_names else None,
            **self._train_params,
        )

        if self._fit_dir:
            self._fit_dir.mkdir(parents=True, exist_ok=True)
            self._booster.save_model(str(self._fit_dir / "model.txt"))

        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        if self._features is None or self._booster is None:
            raise NotFittedError()

        input_data = data.select(self._features).to_pandas()
        pred = self._booster.predict(input_data)
        return pl.from_numpy(pred, schema=self._prediction)  # type: ignore
