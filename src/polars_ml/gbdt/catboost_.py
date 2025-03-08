from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Self

import polars as pl
from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    import catboost as cb


class CatBoost(PipelineComponent):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        params: Mapping[str, Any],
        *,
        cat_features: list[str] | None = None,
        prediction_name: str = "catboost",
        append_prediction: bool = True,
        train_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        predict_kwargs: Mapping[str, Any]
        | Callable[[DataFrame, "cb.CatBoost"], dict[str, Any]]
        | None = None,
        pool_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        save_dir: str | Path | None = None,
    ):
        self.features = features
        self.label = label
        self.params = params
        self.cat_features = cat_features
        self.prediction_name = prediction_name
        self.append_prediction = append_prediction
        self.train_kwargs = train_kwargs or {}
        self.predict_kwargs = predict_kwargs or {}
        self.pool_kwargs = pool_kwargs or {}
        self.save_dir = Path(save_dir) if save_dir is not None else None

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        import catboost as cb

        train_features = data.select(self.features)
        train_label = data.select(self.label)

        pool_kwargs = (
            self.pool_kwargs(data) if callable(self.pool_kwargs) else self.pool_kwargs
        )

        train_pool = cb.Pool(
            data=train_features.to_numpy(),
            label=train_label.to_numpy().squeeze(),
            feature_names=train_features.columns,
            cat_features=self.cat_features,
            **pool_kwargs,
        )

        eval_sets = []
        if validation_data is not None:
            if isinstance(validation_data, DataFrame):
                valid_features = validation_data.select(self.features)
                valid_label = validation_data.select(self.label)

                valid_pool = cb.Pool(
                    data=valid_features.to_numpy(),
                    label=valid_label.to_numpy().squeeze(),
                    feature_names=valid_features.columns,
                    cat_features=self.cat_features,
                    **pool_kwargs,
                )
                eval_sets.append(valid_pool)
            else:
                for raw_valid_data in validation_data.values():
                    valid_features = raw_valid_data.select(self.features)
                    valid_label = raw_valid_data.select(self.label)

                    valid_pool = cb.Pool(
                        data=valid_features.to_numpy(),
                        label=valid_label.to_numpy().squeeze(),
                        feature_names=valid_features.columns,
                        cat_features=self.cat_features,
                        **pool_kwargs,
                    )
                    eval_sets.append(valid_pool)

        train_kwargs = (
            self.train_kwargs(data)
            if callable(self.train_kwargs)
            else self.train_kwargs
        )

        self.model = cb.CatBoost(self.params)
        self.model.fit(
            train_pool,
            eval_set=eval_sets if eval_sets else None,
            **train_kwargs,
        )

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_model(self.save_dir / "model.cbm")

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input = data.select(self.features)
        predict_kwargs = (
            self.predict_kwargs(data, self.model)
            if callable(self.predict_kwargs)
            else self.predict_kwargs
        )

        pred: NDArray[Any] = self.model.predict(input.to_numpy(), **predict_kwargs)

        if pred.ndim == 1:
            schema = [self.prediction_name]
        else:
            schema = [f"{self.prediction_name}_{i}" for i in range(pred.shape[1])]

        pred_df = pl.from_numpy(pred, schema=schema)
        if self.append_prediction:
            return pl.concat([data, pred_df], how="horizontal")
        else:
            return pred_df
