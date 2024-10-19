from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Tuple, override

from polars import DataFrame
from polars._typing import ColumnNameOrSelector

from .x_y_hue_plot import XYHuePlot


class RelationalPlot(XYHuePlot):
    def __init__(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        kind: Literal["scatter", "line"] = "scatter",
        show_progress: bool = True,
        figsize: Tuple[int, int] = (20, 20),
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

        plt.figure(figsize=self.figsize)
        getattr(sns, self.kind + "plot")(data=data, x=x, y=y, hue=hue, **self.plot_kws)

        title = f"{x} vs {y}"
        plt.title(title)
        plt.tight_layout()

        filename = f"x={x}__y={y}__hue={hue}"
        log_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(log_dir / f"{filename}.png")
