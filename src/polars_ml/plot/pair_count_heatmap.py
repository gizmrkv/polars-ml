import itertools
from pathlib import Path
from typing import Any, Iterable, Mapping

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from polars import DataFrame
from polars._typing import ColumnNameOrSelector
from tqdm import tqdm

from polars_ml.pipeline.component import PipelineComponent


class PairCountHeatmapPlot(PipelineComponent):
    def __init__(
        self,
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        show_progress: bool = True,
        subplots_kwargs: Mapping[str, Any] | None = None,
        heatmap_kwargs: Mapping[str, Any] | None = None,
        out_dir: str | Path | None = None,
    ):
        self.features = features
        self.show_progress = show_progress
        self.subplots_kwargs = subplots_kwargs or {"figsize": (10, 10)}
        self.heatmap_kwargs = heatmap_kwargs or {
            "annot": False,
            "square": False,
            "cmap": "viridis",
            "linewidths": 0.0,
        }
        self.out_dir = Path(out_dir) if out_dir else None

    def transform(self, data: DataFrame) -> DataFrame:
        if self.out_dir is None:
            return data

        self.out_dir.mkdir(exist_ok=True, parents=True)

        cats = data.lazy().select(self.features).collect_schema().names()

        for cat1, cat2 in tqdm(
            itertools.combinations(cats, 2),
            total=len(cats) * (len(cats) - 1) // 2,
            disable=not self.show_progress,
        ):
            m = (
                data.group_by(
                    pl.col(cat1).cast(pl.String),
                    pl.col(cat2).cast(pl.String),
                )
                .len()
                .fill_null("")
                .sort(pl.all())
                .pivot(
                    cat2,
                    index=cat1,
                    values="len",
                    sort_columns=True,
                    maintain_order=True,
                )
                .fill_null(0)
            )
            x_labels = m.columns[1:]
            y_labels = m[cat1].to_list()

            fig, ax = plt.subplots(**self.subplots_kwargs)
            sns.heatmap(
                m.drop(cat1).to_numpy(),
                ax=ax,
                xticklabels=x_labels,
                yticklabels=y_labels,
                **self.heatmap_kwargs,
            )
            ax.set_xlabel(cat2)
            ax.set_ylabel(cat1)
            ax.set_title(f"{cat1} vs {cat2}")
            fig.savefig(self.out_dir / f"{cat1} vs {cat2}.png")
            fig.clear()
            plt.close(fig)

        return data
