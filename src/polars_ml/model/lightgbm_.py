from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Self

import polars as pl
from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml import Component

if TYPE_CHECKING:
    import lightgbm as lgb


class LightGBM(Component):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        params: dict[str, Any],
        *,
        prediction_name: str = "lightgbm",
        append_prediction: bool = True,
        train_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        predict_kwargs: dict[str, Any]
        | Callable[[DataFrame, "lgb.Booster"], dict[str, Any]]
        | None = None,
        train_dataset_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        validation_dataset_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        dir: str | Path | None = None,
        plot_importance: bool = False,
    ):
        self.features = features
        self.label = label
        self.params = params
        self.prediction_name = prediction_name
        self.append_prediction = append_prediction
        self.train_kwargs = train_kwargs or {}
        self.predict_kwargs = predict_kwargs or {}
        self.train_dataset_kwargs = train_dataset_kwargs or {}
        self.validation_dataset_kwargs = validation_dataset_kwargs or {}
        self.dir = Path(dir) if dir is not None else None
        self.plot_importance = plot_importance

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        train_data = data.select(self.features)
        train_features = train_data.drop(self.label)
        train_label = train_data[self.label]
        train_dataset_kwargs = (
            self.train_dataset_kwargs(data)
            if callable(self.train_dataset_kwargs)
            else self.train_dataset_kwargs
        )
        train_dataset = lgb.Dataset(
            train_features.to_numpy(),
            label=train_label.to_numpy(),
            feature_name=train_features.columns,
            **train_dataset_kwargs,
        )

        valid_sets = []
        valid_names = []
        if validation_data is not None:
            if isinstance(validation_data, DataFrame):
                valid_data = validation_data.select(self.features)
                valid_features = valid_data.drop(self.label)
                valid_label = valid_data[self.label]
                valid_dataset_kwargs = (
                    self.validation_dataset_kwargs(validation_data)
                    if callable(self.validation_dataset_kwargs)
                    else self.validation_dataset_kwargs
                )
                valid_dataset = train_dataset.create_valid(
                    valid_features.to_numpy(),
                    label=valid_label.to_numpy(),
                    **valid_dataset_kwargs,
                )
                valid_sets.append(valid_dataset)
                valid_names.append("valid")
            else:
                for name, raw_valid_data in validation_data.items():
                    valid_data = raw_valid_data.select(self.features)
                    valid_features = valid_data.drop(self.label)
                    valid_label = valid_data[self.label]
                    valid_dataset_kwargs = (
                        self.validation_dataset_kwargs(raw_valid_data)
                        if callable(self.validation_dataset_kwargs)
                        else self.validation_dataset_kwargs
                    )
                    valid_dataset = train_dataset.create_valid(
                        valid_features.to_numpy(),
                        label=valid_label.to_numpy(),
                        **valid_dataset_kwargs,
                    )
                    valid_sets.append(valid_dataset)
                    valid_names.append(name)

        valid_sets.append(train_dataset)
        valid_names.append("train")

        train_kwargs = (
            self.train_kwargs(data)
            if callable(self.train_kwargs)
            else self.train_kwargs
        )

        self.model = lgb.train(
            self.params,
            train_dataset,
            valid_sets=valid_sets,
            valid_names=valid_names,
            **train_kwargs,
        )

        if self.plot_importance:
            if self.dir is None:
                raise ValueError("dir must be set to plot importance")

            import matplotlib.pyplot as plt

            self.dir.mkdir(parents=True, exist_ok=True)
            for importance_type in ["gain", "split"]:
                lgb.plot_importance(self.model, importance_type=importance_type)
                plt.tight_layout()
                plt.savefig(self.dir / f"importance_{importance_type}.png")
                plt.close()

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input = data.select(self.features).select(pl.exclude(self.label))
        predict_kwargs = (
            self.predict_kwargs(data, self.model)
            if callable(self.predict_kwargs)
            else self.predict_kwargs
        )
        pred: NDArray[Any] = self.model.predict(input.to_numpy(), **predict_kwargs)  # type: ignore
        if pred.ndim == 1:
            schema = [self.prediction_name]
        else:
            schema = [f"{self.prediction_name}_{i}" for i in range(pred.shape[1])]

        pred_df = pl.from_numpy(pred, schema=schema)
        if self.append_prediction:
            return pl.concat([data, pred_df], how="horizontal")
        else:
            return pred_df
