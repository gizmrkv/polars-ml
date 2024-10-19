import itertools
from typing import Any, Dict, Iterable, Tuple, override

from polars import DataFrame
from polars._typing import ColumnNameOrSelector

from polars_ml.component import Component


class ResidualPlot(Component):
    def __init__(
        self,
        y_trues: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        y_preds: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        figsize: Tuple[int, int] = (20, 16),
        scatter_kws: Dict[str, Any] | None = None,
    ):
        self.y_trues = y_trues
        self.y_preds = y_preds
        self.figsize = figsize
        self.scatter_kws = scatter_kws or {}
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

                residuals = y_true - y_pred
                fig, ax = plt.subplots(figsize=self.figsize)
                ax.scatter(y_pred, residuals, **self.scatter_kws)
                ax.axhline(y=0, color="r", linestyle="--")
                ax.set(
                    xlabel="Predicted Values",
                    ylabel="Residuals",
                    title=f"Residual Plot for {y_pred_col} vs {y_true_col}",
                )
                log_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(log_dir / f"y_true={y_true_col}__y_pred={y_pred_col}.png")
                fig.clear()
                plt.close(fig)

        return data
