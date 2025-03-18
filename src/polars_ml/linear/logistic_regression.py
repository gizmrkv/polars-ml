from abc import ABC
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Self, TypedDict

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame, Series
from polars._typing import IntoExpr
from sklearn import linear_model
from sklearn.metrics import auc, roc_curve

from polars_ml.pipeline.component import PipelineComponent


class LogisticRegressionParameters(TypedDict, total=False):
    penalty: Literal["l1", "l2", "elasticnet"] | None
    dual: bool
    tol: float
    C: float
    fit_intercept: bool
    intercept_scaling: float
    class_weight: Mapping[str, float] | str
    random_state: int
    solver: Literal["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
    max_iter: int
    multi_class: Literal["auto", "ovr", "multinomial"]
    verbose: int
    warm_start: bool
    n_jobs: int
    l1_ratio: float


class LogisticRegressionFitArguments(TypedDict, total=False):
    sample_weight: NDArray[Any]


class LogisticRegression(PipelineComponent, ABC):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        *,
        prediction_name: str = "logistic_regression",
        include_input: bool = True,
        predict_proba: bool = False,
        model_kwargs: LogisticRegressionParameters
        | Callable[[DataFrame], LogisticRegressionParameters]
        | None = None,
        fit_kwargs: LogisticRegressionFitArguments
        | Callable[[DataFrame], LogisticRegressionFitArguments]
        | None = None,
        out_dir: str | Path | None = None,
    ):
        self.features = features
        self.label = label
        self.prediction_name = prediction_name
        self.include_input = include_input
        self.predict_proba = predict_proba
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
        self.model = linear_model.LogisticRegression(**model_kwargs)

        X = train_features.to_numpy()
        y = train_label.to_numpy().squeeze()
        fit_kwargs = (
            self.fit_kwargs(data) if callable(self.fit_kwargs) else self.fit_kwargs
        )
        self.model.fit(X, y, **fit_kwargs)

        if self.out_dir is not None:
            y_pred_proba = self.model.predict_proba(X)
            self._save_plots(y, y_pred_proba)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input = data.select(self.features)

        if self.predict_proba:
            pred_proba: NDArray[Any] = self.model.predict_proba(input.to_numpy())
            columns = [
                Series(f"{self.prediction_name}_{name}", pred_proba[:, i])
                for i, name in enumerate(self.model.classes_)
            ]
        else:
            pred_class = self.model.predict(input.to_numpy())
            columns = [Series(self.prediction_name, pred_class)]

        if self.include_input:
            return data.with_columns(columns)
        else:
            return DataFrame(columns)

    def _save_plots(self, y_true: NDArray[Any], y_pred_proba: NDArray[Any]):
        if self.out_dir is None:
            return

        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        self.out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        if y_pred_proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.out_dir / "roc_curve.png")
        plt.close()

        y_pred = np.argmax(y_pred_proba, axis=1)
        if len(self.model.classes_) > 2:
            class_mapping = {i: cls for i, cls in enumerate(self.model.classes_)}
            y_pred = np.array([class_mapping[p] for p in y_pred])

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.out_dir / "confusion_matrix.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        coefficients = (
            self.model.coef_[0]
            if len(self.model.classes_) == 2
            else self.model.coef_.mean(axis=0)
        )
        plt.bar(self.columns, coefficients)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Features")
        plt.ylabel("Coefficient Value")
        plt.title("Feature Coefficients")
        plt.tight_layout()
        plt.savefig(self.out_dir / "feature_coefficients.png")
        plt.close()
