from abc import ABC
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Self, TypedDict

from numpy.typing import NDArray
from polars import DataFrame, Series
from polars._typing import IntoExpr
from sklearn import linear_model

from polars_ml.pipeline.component import PipelineComponent


class LassoParameters(TypedDict, total=False):
    alpha: float
    fit_intercept: bool
    precompute: bool | NDArray[Any]
    max_iter: int
    tol: float
    warm_start: bool
    positive: bool
    random_state: int | None
    selection: Literal["cyclic", "random"]


class LassoFitArguments(TypedDict, total=False):
    sample_weight: NDArray[Any]


class Lasso(PipelineComponent, ABC):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        *,
        prediction_name: str = "lasso",
        include_input: bool = True,
        model_kwargs: LassoParameters
        | Callable[[DataFrame], LassoParameters]
        | None = None,
        fit_kwargs: LassoFitArguments
        | Callable[[DataFrame], LassoFitArguments]
        | None = None,
        out_dir: str | Path | None = None,
    ):
        self.features = features
        self.label = label
        self.prediction_name = prediction_name
        self.include_input = include_input
        self.model_kwargs = model_kwargs or {}
        self.fit_kwargs = fit_kwargs or {}
        self.out_dir = Path(out_dir) if out_dir is not None else None

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        train_features = data.select(self.features)
        train_label = data.select(self.label)
        self.columns = train_features.columns

        model_kwargs = (
            self.model_kwargs(data)
            if callable(self.model_kwargs)
            else self.model_kwargs
        )
        self.model = linear_model.Lasso(copy_X=False, **model_kwargs)

        X = train_features.to_numpy()
        y = train_label.to_numpy().squeeze()
        fit_kwargs = (
            self.fit_kwargs(data) if callable(self.fit_kwargs) else self.fit_kwargs
        )
        self.model.fit(X, y, **fit_kwargs)

        if self.out_dir is not None:
            y_pred = self.model.predict(X)
            self._save_plots(y, y_pred)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input = data.select(self.features)
        pred: NDArray[Any] = self.model.predict(input.to_numpy())

        if self.include_input:
            return data.with_columns(Series(self.prediction_name, pred))
        else:
            return DataFrame(Series(self.prediction_name, pred))

    def _save_plots(self, y_true: NDArray[Any], y_pred: NDArray[Any]):
        if self.out_dir is None:
            return

        import matplotlib.pyplot as plt
        import numpy as np

        self.out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values (Lasso)")
        plt.tight_layout()
        plt.savefig(self.out_dir / "lasso_actual_vs_predicted.png")
        plt.close()

        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot (Lasso)")
        plt.tight_layout()
        plt.savefig(self.out_dir / "lasso_residuals.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        coefficients = self.model.coef_
        coef_importance = np.abs(coefficients)
        sorted_idx = np.argsort(coef_importance)
        pos = np.arange(len(sorted_idx))

        plt.barh(pos, coefficients[sorted_idx])
        plt.yticks(pos, np.array(self.columns)[sorted_idx].tolist())
        plt.xlabel("Coefficient Value")
        plt.title("Feature Coefficients (Lasso)")
        plt.tight_layout()
        plt.savefig(self.out_dir / "lasso_feature_coefficients.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(coefficients[coefficients != 0], bins=20)
        plt.xlabel("Non-zero Coefficient Values")
        plt.ylabel("Frequency")
        plt.title("Distribution of Non-zero Coefficients (Lasso)")
        plt.tight_layout()
        plt.savefig(self.out_dir / "lasso_coefficient_distribution.png")
        plt.close()
