import itertools
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from polars import DataFrame
from polars._typing import ColumnNameOrSelector


class iter_plots:
    def __init__(
        self,
        data: DataFrame,
        *,
        save_dir: str | Path,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        subplots_kwargs: dict[str, Any] | None = None,
    ):
        self.data = data
        self.subplots_kwargs = subplots_kwargs or {}
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.x_columns = (
            [None] if x is None else data.lazy().select(x).collect_schema().names()
        )
        self.y_columns = (
            [] if y is None else data.lazy().select(y).collect_schema().names()
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
            fig, ax = plt.subplots(**self.subplots_kwargs)
            yield self.data, x_col, y_col, hue_col, ax
            filename = "__".join(
                f"{k}={v}"
                for k, v in {"x": x_col, "y": y_col, "hue": hue_col}.items()
                if v is not None
            )
            fig.savefig(self.save_dir / f"{filename}.png")
            fig.clear()
            plt.close(fig)


def count_heatmap(
    data: DataFrame,
    x: str,
    y: str,
    *,
    labels: dict[str, Sequence[Any]] | None = None,
    heatmap_kwargs: dict[str, Any] | None = None,
    ax: plt.Axes,
):
    data = (
        data.select(x, y)
        .group_by(x, y)
        .len()
        .pivot(x, index=y, values="len")
        .fill_null(0)
    )

    labels = labels or {}
    xlabels = labels.get(x, data.columns[1:])
    ylabels = labels.get(y, data[y].to_list())

    m = data.drop(y).to_numpy()
    sns.heatmap(m, ax=ax, **(heatmap_kwargs or {}))
    ax.set_xticklabels(xlabels, rotation=90)
    ax.set_yticklabels(ylabels, rotation=0)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
