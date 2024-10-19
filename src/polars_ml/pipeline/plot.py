from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterable, Literal, Tuple

from polars._typing import ColumnNameOrSelector

from polars_ml.plot import (
    CategoricalIndependencyMatrix,
    CategoricalPlot,
    CorrelationMatrix,
    DistributionPlot,
    NullMatrix,
    PRCurve,
    RelationalPlot,
    ResidualPlot,
    ROCCurve,
)
from polars_ml.typing import PipelineType

if TYPE_CHECKING:
    from .lazy_pipeline import LazyPipeline  # noqa: F401
    from .pipeline import Pipeline  # noqa: F401


class BasePlotNameSpace(Generic[PipelineType], ABC):
    def __init__(self, pipeline: PipelineType):
        self.pipeline = pipeline


class PlotNameSpace(BasePlotNameSpace["Pipeline"]):
    def corr(
        self,
        *,
        figsize: Tuple[int, int] = (22, 20),
        heatmap_kws: Dict[str, Any] | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            CorrelationMatrix(figsize=figsize, heatmap_kws=heatmap_kws)
        )

    def null(
        self,
        *,
        figsize: Tuple[int, int] = (20, 20),
        heatmap_kws: Dict[str, Any] | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(NullMatrix(figsize=figsize, heatmap_kws=heatmap_kws))

    def categorical(
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
        return self.pipeline.pipe(
            CategoricalPlot(
                x=x,
                y=y,
                hue=hue,
                kind=kind,
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            )
        )

    def strip(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            CategoricalPlot(
                x=x,
                y=y,
                hue=hue,
                kind="strip",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            ).set_component_name("StripPlot")
        )

    def swarm(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            CategoricalPlot(
                x=x,
                y=y,
                hue=hue,
                kind="swarm",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            ).set_component_name("SwarmPlot")
        )

    def box(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            CategoricalPlot(
                x=x,
                y=y,
                hue=hue,
                kind="box",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            ).set_component_name("BoxPlot")
        )

    def violin(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            CategoricalPlot(
                x=x,
                y=y,
                hue=hue,
                kind="violin",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            ).set_component_name("ViolinPlot")
        )

    def boxen(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            CategoricalPlot(
                x=x,
                y=y,
                hue=hue,
                kind="boxen",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            ).set_component_name("BoxenPlot")
        )

    def point(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            CategoricalPlot(
                x=x,
                y=y,
                hue=hue,
                kind="point",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            ).set_component_name("PointPlot")
        )

    def bar(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            CategoricalPlot(
                x=x,
                y=y,
                hue=hue,
                kind="bar",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            ).set_component_name("BarPlot")
        )

    def count(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            CategoricalPlot(
                x=x,
                y=y,
                hue=hue,
                kind="count",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            ).set_component_name("CountPlot")
        )

    def distribution(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        kind: Literal["hist", "kde", "ecdf"] = "hist",
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            DistributionPlot(
                x=x,
                y=y,
                hue=hue,
                kind=kind,
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            )
        )

    def hist(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            DistributionPlot(
                x=x,
                y=y,
                hue=hue,
                kind="hist",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            ).set_component_name("HistPlot")
        )

    def kde(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            DistributionPlot(
                x=x,
                y=y,
                hue=hue,
                kind="kde",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            ).set_component_name("KDEPlot")
        )

    def ecdf(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            DistributionPlot(
                x=x,
                y=y,
                hue=hue,
                kind="ecdf",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            ).set_component_name("ECDFPlot")
        )

    def relational(
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
        return self.pipeline.pipe(
            RelationalPlot(
                x=x,
                y=y,
                hue=hue,
                kind=kind,
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            )
        )

    def scatter(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: Tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            RelationalPlot(
                x=x,
                y=y,
                hue=hue,
                kind="scatter",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws or {"s": 12, "linewidth": 0},
            ).set_component_name("ScatterPlot")
        )

    def line(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
        figsize: Tuple[int, int] = (20, 20),
        plot_kws: Dict[str, Any] | None = None,
    ):
        return self.pipeline.pipe(
            RelationalPlot(
                x=x,
                y=y,
                hue=hue,
                kind="line",
                show_progress=show_progress,
                figsize=figsize,
                plot_kws=plot_kws,
            ).set_component_name("LinePlot")
        )

    def categorical_independency_matrix(
        self,
        *,
        figsize: Tuple[int, int] = (11, 9),
        heatmap_kws: Dict[str, Any] | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            CategoricalIndependencyMatrix(figsize=figsize, heatmap_kws=heatmap_kws)
        )

    def roc_curve(
        self,
        y_trues: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        y_preds: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        figsize: Tuple[int, int] = (20, 16),
    ) -> "Pipeline":
        return self.pipeline.pipe(
            ROCCurve(y_trues=y_trues, y_preds=y_preds, figsize=figsize)
        )

    def pr_curve(
        self,
        y_trues: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        y_preds: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        figsize: Tuple[int, int] = (20, 16),
    ) -> "Pipeline":
        return self.pipeline.pipe(
            PRCurve(y_trues=y_trues, y_preds=y_preds, figsize=figsize)
        )

    def residual(
        self,
        y_trues: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        y_preds: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        figsize: Tuple[int, int] = (20, 16),
        scatter_kws: Dict[str, Any] | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            ResidualPlot(
                y_trues=y_trues,
                y_preds=y_preds,
                figsize=figsize,
                scatter_kws=scatter_kws,
            )
        )


class LazyPlotNameSpace(BasePlotNameSpace["LazyPipeline"]):
    pass
