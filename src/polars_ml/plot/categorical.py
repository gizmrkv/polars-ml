import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, override

from polars import DataFrame
from polars._typing import ColumnNameOrSelector

from .x_y_hue_plot import XYHuePlot


class CategoricalPlot(XYHuePlot):
    def __init__(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        kind: Literal[
            "strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"
        ] = "box",
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        super().__init__(show_progress=show_progress)
        self.x = x
        self.y = y
        self.hue = hue
        self.kind = kind
        self.figsize = figsize
        self.plot_kws = plot_kws or {}
        self._is_fitted = True

    @override
    def plot(
        self,
        data: DataFrame,
        x: str | None,
        y: str | None,
        hue: str | None,
        *,
        log_dir: Path,
    ):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if "order" in self.plot_kws:
            order = self.plot_kws["order"]
            plot_kws = self.plot_kws.copy()
            plot_kws.pop("order")
        elif self.kind == "count":
            if x is not None:
                column = x
            elif y is not None:
                column = y
            else:
                raise ValueError("Either x or y must be provided")

            len_index = uuid.uuid4().hex
            order = (
                data.group_by(column)
                .len(len_index)
                .sort(len_index, descending=True)
                .drop(len_index)[column]
            )
            plot_kws = self.plot_kws.copy()
        else:
            cols = [c for c in [x, y, hue] if c is not None]
            data = data.select(cols).sort(cols)
            order = None
            plot_kws = self.plot_kws.copy()

        plt.figure(figsize=self.figsize)
        getattr(sns, self.kind + "plot")(
            data=data, x=x, y=y, hue=hue, order=order, **plot_kws
        )

        title = "Count" if self.kind == "count" else str(x)
        if hue is not None:
            title += f" by {hue}"
        plt.title(title)
        plt.tight_layout()

        filename = f"x={x}__y={y}__hue={hue}"
        log_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(log_dir / f"{filename}.png")
