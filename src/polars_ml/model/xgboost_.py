from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Self

import polars as pl
from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml import Component

if TYPE_CHECKING:
    import xgboost as xgb


class XGBoost(Component):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        params: dict[str, Any],
        *,
        prediction_name: str = "xgboost",
        append_prediction: bool = True,
        train_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        predict_kwargs: dict[str, Any]
        | Callable[[DataFrame, "xgb.Booster"], dict[str, Any]]
        | None = None,
        train_dmatrix_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        validation_dmatrix_kwargs: dict[str, Any]
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
        self.train_dmatrix_kwargs = train_dmatrix_kwargs or {}
        self.validation_dmatrix_kwargs = validation_dmatrix_kwargs or {}
        self.dir = Path(dir) if dir is not None else None
        self.plot_importance = plot_importance

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        import xgboost as xgb

        train_features = data.select(self.features)
        train_label = data.select(self.label)

        train_dmatrix_kwargs = (
            self.train_dmatrix_kwargs(data)
            if callable(self.train_dmatrix_kwargs)
            else self.train_dmatrix_kwargs
        )
        train_dataset = xgb.DMatrix(
            train_features.to_numpy(),
            label=train_label.to_numpy().squeeze(),
            feature_names=train_features.columns,
            **train_dmatrix_kwargs,
        )

        evals = [(train_dataset, "train")]
        if validation_data is not None:
            if isinstance(validation_data, DataFrame):
                valid_features = validation_data.select(self.features)
                valid_label = validation_data.select(self.label)

                valid_dmatrix_kwargs = (
                    self.validation_dmatrix_kwargs(validation_data)
                    if callable(self.validation_dmatrix_kwargs)
                    else self.validation_dmatrix_kwargs
                )
                valid_dataset = xgb.DMatrix(
                    valid_features.to_numpy(),
                    label=valid_label.to_numpy().squeeze(),
                    feature_names=valid_features.columns,
                    **valid_dmatrix_kwargs,
                )
                evals.append((valid_dataset, "valid"))
            else:
                for name, raw_valid_data in validation_data.items():
                    valid_features = raw_valid_data.select(self.features)
                    valid_label = raw_valid_data.select(self.label)

                    valid_dmatrix_kwargs = (
                        self.validation_dmatrix_kwargs(raw_valid_data)
                        if callable(self.validation_dmatrix_kwargs)
                        else self.validation_dmatrix_kwargs
                    )
                    valid_dataset = xgb.DMatrix(
                        valid_features.to_numpy(),
                        label=valid_label.to_numpy().squeeze(),
                        feature_names=valid_features.columns,
                        **valid_dmatrix_kwargs,
                    )
                    evals.append((valid_dataset, name))

        train_kwargs = (
            self.train_kwargs(data)
            if callable(self.train_kwargs)
            else self.train_kwargs
        )

        self.model = xgb.train(
            self.params,
            train_dataset,
            evals=evals,
            **train_kwargs,
        )

        if self.plot_importance:
            if self.dir is None:
                raise ValueError("dir must be set to plot importance")

            import matplotlib.pyplot as plt

            self.dir.mkdir(parents=True, exist_ok=True)
            for importance_type in ["weight", "gain", "cover"]:
                xgb.plot_importance(self.model, importance_type=importance_type)
                plt.tight_layout()
                plt.savefig(self.dir / f"importance_{importance_type}.png")
                plt.close()

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input = data.select(self.features)
        predict_kwargs = (
            self.predict_kwargs(data, self.model)
            if callable(self.predict_kwargs)
            else self.predict_kwargs
        )

        dtest = xgb.DMatrix(input.to_numpy(), feature_names=input.columns)
        pred: NDArray[Any] = self.model.predict(dtest, **predict_kwargs)

        if pred.ndim == 1:
            schema = [self.prediction_name]
        else:
            schema = [f"{self.prediction_name}_{i}" for i in range(pred.shape[1])]

        pred_df = pl.from_numpy(pred, schema=schema)
        if self.append_prediction:
            return pl.concat([data, pred_df], how="horizontal")
        else:
            return pred_df
