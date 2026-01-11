import logging
import time
from typing import Self

from polars import DataFrame

from polars_ml.base import HasFeatureImportance, Transformer
from polars_ml.gbdt import GBDTNameSpace
from polars_ml.linear import LinearNameSpace
from polars_ml.metrics import MetricsNameSpace
from polars_ml.optimize import OptimizeNameSpace
from polars_ml.preprocessing.horizontal import HorizontalNameSpace

from .mixin import PipelineMixin


class Pipeline(PipelineMixin, HasFeatureImportance):
    def __init__(
        self,
        *steps: Transformer,
        verbose: bool | logging.Logger = False,
    ):
        self.steps: list[Transformer] = list(steps)
        if isinstance(verbose, logging.Logger):
            self.verbose = True
            self.logger = verbose
        else:
            self.verbose = verbose
            self.logger = logging.getLogger(__name__)

    def pipe(self, step: Transformer) -> Self:
        self.steps.append(step)
        return self

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f}s"

        s = seconds % 60
        m = int(seconds // 60)
        h = int(m // 60)
        m = m % 60
        d = int(h // 24)
        h = h % 24

        parts = []
        if d > 0:
            parts.append(f"{d}d")
        if h > 0 or d > 0:
            parts.append(f"{h:02d}h")
        if m > 0 or h > 0 or d > 0:
            parts.append(f"{m:02d}m")
        parts.append(f"{s:05.2f}s")

        return " ".join(parts)

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        for i, step in enumerate(self.steps):
            step_name = type(step).__name__
            if self.verbose:
                self.logger.info(f"[{i + 1}/{len(self.steps)}] {step_name} fitting...")
            start_time = time.perf_counter()
            try:
                if i < len(self.steps) - 1:
                    data = step.fit_transform(data, **more_data)
                    more_data = {k: step.transform(v) for k, v in more_data.items()}
                else:
                    step.fit(data, **more_data)

                elapsed = time.perf_counter() - start_time
                if self.verbose:
                    self.logger.info(
                        f"[{i + 1}/{len(self.steps)}] {step_name} finished in {self._format_time(elapsed)}"
                    )
            except Exception as e:
                if e.args and isinstance(e.args[0], str):
                    e.args = (f"Step {i} ({step_name}): {e.args[0]}",) + e.args[1:]
                else:
                    e.args = (f"Step {i} ({step_name})",) + e.args
                raise
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        for i, step in enumerate(self.steps):
            step_name = type(step).__name__
            if self.verbose:
                self.logger.info(
                    f"[{i + 1}/{len(self.steps)}] {step_name} fit-transforming..."
                )
            start_time = time.perf_counter()
            try:
                data = step.fit_transform(data, **more_data)
                if i < len(self.steps) - 1:
                    more_data = {k: step.transform(v) for k, v in more_data.items()}

                elapsed = time.perf_counter() - start_time
                if self.verbose:
                    self.logger.info(
                        f"[{i + 1}/{len(self.steps)}] {step_name} finished in {self._format_time(elapsed)}"
                    )
            except Exception as e:
                if e.args and isinstance(e.args[0], str):
                    e.args = (f"Step {i} ({step_name}): {e.args[0]}",) + e.args[1:]
                else:
                    e.args = (f"Step {i} ({step_name})",) + e.args
                raise
        return data

    def transform(self, data: DataFrame) -> DataFrame:
        for i, step in enumerate(self.steps):
            step_name = type(step).__name__
            if self.verbose:
                self.logger.info(
                    f"[{i + 1}/{len(self.steps)}] {step_name} transforming..."
                )
            start_time = time.perf_counter()
            try:
                data = step.transform(data)
                elapsed = time.perf_counter() - start_time
                if self.verbose:
                    self.logger.info(
                        f"[{i + 1}/{len(self.steps)}] {step_name} finished in {self._format_time(elapsed)}"
                    )
            except Exception as e:
                if e.args and isinstance(e.args[0], str):
                    e.args = (f"Step {i} ({step_name}): {e.args[0]}",) + e.args[1:]
                else:
                    e.args = (f"Step {i} ({step_name})",) + e.args
                raise
        return data

    def get_feature_importance(self) -> DataFrame:
        if not self.steps:
            raise ValueError("Pipeline has no steps.")

        last_step = self.steps[-1]
        if isinstance(last_step, HasFeatureImportance):
            return last_step.get_feature_importance()

        raise TypeError(
            f"The last step of the pipeline ({type(last_step).__name__}) "
            "does not support feature importance."
        )

    @property
    def gbdt(self) -> GBDTNameSpace:
        return GBDTNameSpace(self)

    @property
    def linear(self) -> LinearNameSpace:
        return LinearNameSpace(self)

    @property
    def metrics(self) -> MetricsNameSpace:
        return MetricsNameSpace(self)

    @property
    def optimize(self) -> OptimizeNameSpace:
        return OptimizeNameSpace(self)

    @property
    def horizontal(self) -> HorizontalNameSpace:
        return HorizontalNameSpace(self)
