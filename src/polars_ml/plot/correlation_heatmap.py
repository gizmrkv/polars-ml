from pathlib import Path
from typing import Any, Iterable, Mapping

import seaborn as sns
from matplotlib import pyplot as plt
from polars import DataFrame
from polars._typing import ColumnNameOrSelector

from polars_ml.pipeline.component import PipelineComponent


class CorrelationHeatmapPlot(PipelineComponent):
    def __init__(
        self,
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        subplots_kwargs: Mapping[str, Any] | None = None,
        heatmap_kwargs: Mapping[str, Any] | None = None,
        out_dir: str | Path = "correlation_heatmap",
    ):
        self.features = features
        self.subplots_kwargs = subplots_kwargs or {"figsize": (10, 10)}
        self.heatmap_kwargs = heatmap_kwargs or {
            "annot": True,
            "square": True,
            "cmap": "coolwarm",
            "center": 0,
            "linewidths": 0.5,
        }
        self.out_dir = Path(out_dir)

    def transform(self, data: DataFrame) -> DataFrame:
        self.out_dir.mkdir(exist_ok=True, parents=True)
        d = data.select(self.features)
        m = d.corr().to_numpy()
        labels = d.columns
        fig, ax = plt.subplots(**self.subplots_kwargs)
        sns.heatmap(
            m, xticklabels=labels, yticklabels=labels, ax=ax, **self.heatmap_kwargs
        )
        fig.savefig(self.out_dir / "corr_heatmap.png")
        fig.clear()
        plt.close(fig)

        return data
