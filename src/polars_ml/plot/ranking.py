import itertools
from typing import Iterable, Tuple, override

from polars import DataFrame
from polars._typing import ColumnNameOrSelector
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from polars_ml.component import Component


class ROCCurve(Component):
    def __init__(
        self,
        y_trues: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        y_preds: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        figsize: Tuple[int, int] = (20, 16),
    ):
        self.y_trues = y_trues
        self.y_preds = y_preds
        self.figsize = figsize
        self._is_fitted = True

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        if log_dir := self.log_dir:
            import matplotlib.pyplot as plt

            y_true_columns = data.lazy().select(self.y_trues).collect_schema().names()
            y_pred_columns = data.lazy().select(self.y_preds).collect_schema().names()

            for y_true_col, y_pred_col in itertools.product(
                y_true_columns, y_pred_columns
            ):
                y_true = data[y_true_col].to_numpy()
                y_pred = data[y_pred_col].to_numpy()
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                roc_auc = roc_auc_score(y_true, y_pred)

                fig, ax = plt.subplots(figsize=self.figsize)
                ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
                ax.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                    lw=2,
                    color="r",
                    label="Random guess",
                )
                ax.set(
                    xlabel="False Positive Rate",
                    ylabel="True Positive Rate",
                    title=f"ROC Curve for {y_pred_col} vs {y_true_col}",
                )
                ax.legend()
                log_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(log_dir / f"y_true={y_true_col}__y_pred={y_pred_col}.png")
                fig.clear()
                plt.close(fig)

        return data


class PRCurve(Component):
    def __init__(
        self,
        y_trues: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        y_preds: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        figsize: Tuple[int, int] = (20, 16),
    ):
        self.y_trues = y_trues
        self.y_preds = y_preds
        self.figsize = figsize
        self._is_fitted = True

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        if log_dir := self.log_dir:
            import matplotlib.pyplot as plt

            y_true_columns = data.lazy().select(self.y_trues).collect_schema().names()
            y_pred_columns = data.lazy().select(self.y_preds).collect_schema().names()

            for y_true_col, y_pred_col in itertools.product(
                y_true_columns, y_pred_columns
            ):
                y_true = data[y_true_col].to_numpy()
                y_pred = data[y_pred_col].to_numpy()
                precision, recall, _ = precision_recall_curve(y_true, y_pred)
                pr_auc = auc(recall, precision)

                fig, ax = plt.subplots(figsize=self.figsize)
                ax.plot(recall, precision, label=f"PR curve (area = {pr_auc:.2f})")
                ax.set(
                    xlabel="Recall",
                    ylabel="Precision",
                    title=f"PR Curve for {y_pred_col} vs {y_true_col}",
                )
                ax.legend()
                log_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(log_dir / f"y_true={y_true_col}__y_pred={y_pred_col}.png")
                fig.clear()
                plt.close(fig)

        return data
