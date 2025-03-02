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
        label: IntoExpr,
        params: Mapping[str, Any],
        *,
        prediction_name: str = "lightgbm",
        append_prediction: bool = True,
        train_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        predict_kwargs: Mapping[str, Any]
        | Callable[[DataFrame, "lgb.Booster"], dict[str, Any]]
        | None = None,
        train_dataset_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        validation_dataset_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        save_dir: str | Path | None = None,
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
        self.save_dir = Path(save_dir) if save_dir is not None else None

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        import lightgbm as lgb

        train_features = data.select(self.features)
        train_label = data.select(self.label)
        train_dataset_kwargs = (
            self.train_dataset_kwargs(data)
            if callable(self.train_dataset_kwargs)
            else self.train_dataset_kwargs
        )
        train_dataset = lgb.Dataset(
            train_features.to_numpy(),
            label=train_label.to_numpy().squeeze(),
            feature_name=train_features.columns,
            **train_dataset_kwargs,
        )

        valid_sets = []
        valid_names = []
        if validation_data is not None:
            if isinstance(validation_data, DataFrame):
                valid_features = validation_data.select(self.features)
                valid_label = validation_data.select(self.label)

                valid_dataset_kwargs = (
                    self.validation_dataset_kwargs(validation_data)
                    if callable(self.validation_dataset_kwargs)
                    else self.validation_dataset_kwargs
                )
                valid_dataset = train_dataset.create_valid(
                    valid_features.to_numpy(),
                    label=valid_label.to_numpy().squeeze(),
                    **valid_dataset_kwargs,
                )
                valid_sets.append(valid_dataset)
                valid_names.append("valid")
            else:
                for name, raw_valid_data in validation_data.items():
                    valid_features = raw_valid_data.select(self.features)
                    valid_label = raw_valid_data.select(self.label)
                    valid_dataset_kwargs = (
                        self.validation_dataset_kwargs(raw_valid_data)
                        if callable(self.validation_dataset_kwargs)
                        else self.validation_dataset_kwargs
                    )
                    valid_dataset = train_dataset.create_valid(
                        valid_features.to_numpy(),
                        label=valid_label.to_numpy().squeeze(),
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
            dict(**self.params),
            train_dataset,
            valid_sets=valid_sets,
            valid_names=valid_names,
            **train_kwargs,
        )

        if self.save_dir is not None:
            import matplotlib.pyplot as plt

            self.save_dir.mkdir(parents=True, exist_ok=True)

            self.model.save_model(self.save_dir / "model.txt")

            for importance_type in ["gain", "split"]:
                lgb.plot_importance(self.model, importance_type=importance_type)
                plt.tight_layout()
                plt.savefig(self.save_dir / f"importance_{importance_type}.png")
                plt.close()

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input = data.select(self.features)
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
