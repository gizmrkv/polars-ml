import itertools
from typing import Iterable, Tuple

from polars import DataFrame
from polars._typing import ColumnNameOrSelector
from sklearn.metrics import confusion_matrix

from polars_ml.component import Component


class ConfusionMatrix(Component):
    def __init__(
        self,
        y_trues: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        y_preds: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        class_names: Iterable[str],
        figsize: Tuple[int, int] = (20, 16),
    ):
        self.y_trues = y_trues
        self.y_preds = y_preds
        self.class_names = list(class_names)
        self.figsize = figsize
        self._is_fitted = True

    def execute2(self, data: DataFrame) -> DataFrame:
        if log_dir := self.log_dir:
            import matplotlib.pyplot as plt
            import seaborn as sns

            y_true_columns = data.lazy().select(self.y_trues).collect_schema().names()
            y_pred_columns = data.lazy().select(self.y_preds).collect_schema().names()

            for y_true_col, y_pred_col in itertools.product(
                y_true_columns, y_pred_columns
            ):
                y_true = data[y_true_col].to_numpy()
                y_pred = data[y_pred_col].to_numpy()

                cm = confusion_matrix(y_true, y_pred)

                fig, ax = plt.subplots(figsize=self.figsize)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="viridis",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=ax,
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title(f"Confusion Matrix for {y_pred_col} vs {y_true_col}")
                log_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(log_dir / f"y_true={y_true_col}__y_pred={y_pred_col}.png")
                fig.clear()
                plt.close(fig)

        return data
