from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from numpy.typing import NDArray
from polars._typing import ColumnNameOrSelector

from ..preprocessing.polynomial import Polynomial
from .openfe_ import OpenFE

if TYPE_CHECKING:
    from polars_ml import Pipeline

__all__ = ["Polynomial", "OpenFE"]


class FeatureEngineeringNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def polynomial(
        self,
        *features: ColumnNameOrSelector,
        degree: int = 2,
        component_name: str | None = None,
    ) -> "Pipeline":
        self.pipeline.pipe(
            Polynomial(*features, degree=degree), component_name=component_name
        )
        return self.pipeline

    def openfe(
        self,
        label: str,
        *,
        max_order: int = 1,
        numerical_features: Sequence[str],
        categorical_features: Sequence[str],
        n_subsamples: int = 8,
        params_stage_1: Mapping[str, Any],
        init_score: str | Sequence[str],
        metric_fn: Callable[[NDArray[Any], NDArray[Any]], float],
        is_higher_better: bool = True,
        halving_ratio: float = 0.5,
        min_candidates: int = 2000,
        params_stage_2: Mapping[str, Any],
        n_best_features: int = 100,
        save_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        self.pipeline.pipe(
            OpenFE(
                label,
                max_order=max_order,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                n_subsamples=n_subsamples,
                params_stage_1=params_stage_1,
                init_score=init_score,
                metric_fn=metric_fn,
                is_higher_better=is_higher_better,
                halving_ratio=halving_ratio,
                min_candidates=min_candidates,
                params_stage_2=params_stage_2,
                n_best_features=n_best_features,
                save_dir=save_dir,
            ),
            component_name=component_name,
        )
        return self.pipeline
