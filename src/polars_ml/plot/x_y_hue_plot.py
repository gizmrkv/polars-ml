import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Tuple, override

from polars import DataFrame
from polars._typing import ColumnNameOrSelector

from polars_ml.component import Component


class XYHuePlot(Component, ABC):
    def __init__(
        self,
        x: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        y: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        hue: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        show_progress: bool = True,
    ):
        self.x = x
        self.y = y
        self.hue = hue
        self.show_progress = show_progress
        self._is_fitted = True

    def x_y_hue(
        self, data: DataFrame
    ) -> List[Tuple[str | None, str | None, str | None]]:
        xs = (
            [None]
            if self.x is None
            else data.lazy().select(self.x).collect_schema().names()
        )
        ys = (
            [None]
            if self.y is None
            else data.lazy().select(self.y).collect_schema().names()
        )
        hues = (
            [None]
            if self.hue is None
            else data.lazy().select(self.hue).collect_schema().names()
        )
        return [
            (x, y, hue)
            for x, y, hue in itertools.product(xs, ys, hues)
            if len(set([x, y, hue])) == 3
            or len([c for c in [x, y, hue] if c is None]) == 2
        ]

    @abstractmethod
    def plot(
        self,
        data: DataFrame,
        x: str | None,
        y: str | None,
        hue: str | None,
        *,
        log_dir: Path,
    ): ...

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        if log_dir := self.log_dir:
            import matplotlib.pyplot as plt
            from tqdm import tqdm

            progress_bar = tqdm(
                [(x, y, hue) for x, y, hue in self.x_y_hue(data)],
                desc=self.component_name,
                disable=not self.show_progress,
            )
            for x, y, hue in progress_bar:
                postfix_str = ", ".join(
                    f"{k}={v}"
                    for k, v in [("x", x), ("y", y), ("hue", hue)]
                    if v is not None
                )
                progress_bar.set_description(postfix_str)
                self.plot(data, x, y, hue, log_dir=log_dir)
                plt.clf()
                plt.close()

        return data
