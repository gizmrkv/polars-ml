from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from polars._typing import ColumnNameOrSelector

from .correlation_heatmap import CorrelationHeatmapPlot
from .histogram_plot import HistogramPlot
from .pair_count_heatmap import PairCountHeatmapPlot
from .scatter_plot import ScatterPlot

if TYPE_CHECKING:
    from polars_ml import Pipeline
__all__ = [
    "CorrelationHeatmapPlot",
    "PairCountHeatmapPlot",
    "HistogramPlot",
    "ScatterPlot",
]


class PlotNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def corr(
        self,
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        subplots_kwargs: Mapping[str, Any] | None = None,
        heatmap_kwargs: Mapping[str, Any] | None = None,
        out_dir: str | Path = "correlation_heatmap",
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            CorrelationHeatmapPlot(
                features=features,
                subplots_kwargs=subplots_kwargs,
                heatmap_kwargs=heatmap_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )

    def pair_count(
        self,
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        show_progress: bool = True,
        subplots_kwargs: Mapping[str, Any] | None = None,
        heatmap_kwargs: Mapping[str, Any] | None = None,
        out_dir: str | Path = "pair_count_heatmap",
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            PairCountHeatmapPlot(
                features=features,
                show_progress=show_progress,
                subplots_kwargs=subplots_kwargs,
                heatmap_kwargs=heatmap_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )

    def histogram(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        subplots_kwargs: Mapping[str, Any] | None = None,
        histogram_plot_kwargs: Mapping[str, Any] | None = None,
        out_dir: str | Path = "histogram_plot",
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            HistogramPlot(
                x=x,
                hue=hue,
                show_progress=show_progress,
                subplots_kwargs=subplots_kwargs,
                histogram_plot_kwargs=histogram_plot_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )

    def scatter(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        subplots_kwargs: Mapping[str, Any] | None = None,
        scatter_plot_kwargs: Mapping[str, Any] | None = None,
        out_dir: str | Path = "scatter_plot",
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            ScatterPlot(
                x=x,
                y=y,
                hue=hue,
                show_progress=show_progress,
                subplots_kwargs=subplots_kwargs,
                scatter_plot_kwargs=scatter_plot_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )
