import itertools
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
from polars import DataFrame
from polars._typing import ColumnNameOrSelector


class iter_plots:
    def __init__(
        self,
        data: DataFrame,
        *,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        figsize: tuple[float, float] | None = None,
        save_dir: str | Path,
    ):
        self.data = data
        self.figsize = figsize
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.x_columns = (
            [None] if x is None else data.lazy().select(x).collect_schema().names()
        )
        self.y_columns = (
            [None] if y is None else data.lazy().select(y).collect_schema().names()
        )
        self.hue_columns = (
            [None] if hue is None else data.lazy().select(hue).collect_schema().names()
        )

    def __len__(self):
        return len(self.x_columns) * len(self.y_columns) * len(self.hue_columns)

    def __iter__(
        self,
    ) -> Iterator[tuple[DataFrame, str | None, str | None, str | None, plt.Axes]]:
        for x_col, y_col, hue_col in itertools.product(
            self.x_columns, self.y_columns, self.hue_columns
        ):
            fig, ax = plt.subplots(figsize=self.figsize)
            yield self.data, x_col, y_col, hue_col, ax
            filename = "__".join(
                f"{k}={v}"
                for k, v in {"x": x_col, "y": y_col, "hue": hue_col}.items()
                if v is not None
            )
            fig.savefig(self.save_dir / f"{filename}.png")
            fig.clear()
            plt.close(fig)
