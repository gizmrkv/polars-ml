import itertools
from pathlib import Path
from typing import Any, Iterable, Mapping

import seaborn as sns
from matplotlib import pyplot as plt
from polars import DataFrame
from polars._typing import ColumnNameOrSelector
from tqdm import tqdm

from polars_ml.pipeline.component import PipelineComponent


class HistogramPlot(PipelineComponent):
    def __init__(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        subplots_kwargs: Mapping[str, Any] | None = None,
        histogram_plot_kwargs: Mapping[str, Any] | None = None,
        out_dir: str | Path | None = None,
    ):
        self.x = x
        self.hue = hue
        self.show_progress = show_progress
        self.subplots_kwargs = subplots_kwargs or {"figsize": (10, 6)}
        self.histogram_plot_kwargs = histogram_plot_kwargs or {
            "alpha": 0.5,
            "kde": True,
            "element": "step",
        }
        self.out_dir = Path(out_dir) if out_dir else None

    def transform(self, data: DataFrame) -> DataFrame:
        if self.out_dir is None:
            return data

        self.out_dir.mkdir(exist_ok=True, parents=True)

        xs = data.lazy().select(self.x).collect_schema().names()

        if self.hue is not None:
            hues = data.lazy().select(self.hue).collect_schema().names()
        else:
            hues = [None]

        for x, hue in tqdm(
            list(itertools.product(xs, hues)), disable=not self.show_progress
        ):
            if len(set([x, hue])) < 2:
                continue

            fig, ax = plt.subplots(**self.subplots_kwargs)
            tmp = data.select(set([x, hue]) if hue else x)

            sns.histplot(tmp, x=x, hue=hue, ax=ax, **self.histogram_plot_kwargs)

            plt.xlabel(x)
            plt.ylabel("Count")
            title = x + (f" by {hue}" if hue else "")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(self.out_dir / f"{title}.png")
            fig.clear()
            plt.close()

        return data
