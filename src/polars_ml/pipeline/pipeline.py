from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, Sequence, override

from polars import DataFrame
from polars._typing import (
    AvroCompression,
    ColumnNameOrSelector,
    CsvQuoteStyle,
    ParquetCompression,
)
from tqdm import tqdm

from polars_ml.component import Component, LazyComponent
from polars_ml.utils import Display, GetAttr, Lazy, Print, Write

from .base_pipeline import BasePipeline
from .horizontal import HorizontalNameSpace
from .plot import PlotNameSpace
from .split import SplitNameSpace
from .stat import StatNameSpace
from .transform import TransformNameSpace

if TYPE_CHECKING:
    from .lazy_pipeline import LazyPipeline


class Pipeline(BasePipeline, Component):
    def __init__(
        self,
        *,
        log_dir: str | Path | None = None,
        pipeline_name: str | None = None,
        show_progress: bool = False,
    ):
        super().__init__()

        if log_dir := log_dir:
            self.set_log_dir(log_dir)
        if pipeline_name := pipeline_name:
            self.set_component_name(pipeline_name)

        self.show_progress = show_progress

    @staticmethod
    def load(path: Path) -> "Pipeline":
        import joblib

        pipe = joblib.load(path)
        if isinstance(pipe, Pipeline):
            return pipe
        else:
            raise ValueError(f"Expected a LazyPipeline but got {type(pipe)}")

    def pipe(self, *components: Component | LazyComponent) -> Self:
        self.components.extend(
            Lazy(components) if isinstance(components, Component) else components
            for components in components
        )
        return self

    @override
    def fit(self, data: DataFrame) -> Self:
        bar = tqdm(total=len(self.components), disable=not self.show_progress)
        for i, component in enumerate(self.components):
            bar.set_description(component.component_name)
            if i == len(self.components) - 1:
                component.fit(data.lazy())
            else:
                data = component.fit_execute(data.lazy()).collect()
            bar.update()

        return self

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        bar = tqdm(total=len(self.components), disable=not self.show_progress)
        for component in self.components:
            bar.set_description(component.component_name)
            data = component.execute(data.lazy()).collect()
            bar.update()

        return data

    @override
    def fit_execute(self, data: DataFrame) -> DataFrame:
        bar = tqdm(total=len(self.components), disable=not self.show_progress)
        for component in self.components:
            bar.set_description(component.component_name)
            data = component.fit_execute(data.lazy()).collect()
            bar.update()

        return data

    def lazy(self) -> "LazyPipeline":
        return LazyPipeline(
            log_dir=self.log_dir, show_progress=self.show_progress
        ).pipe(Lazy(self))

    def to_dummies(
        self,
        columns: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        *,
        separator: str = "_",
        drop_first: bool = False,
    ) -> "Pipeline":
        return self.pipe(
            GetAttr(
                "to_dummies",
                columns,
                separator=separator,
                drop_first=drop_first,
            )
        )

    def write_csv(
        self,
        filename: str,
        *,
        include_bom: bool = False,
        include_header: bool = True,
        separator: str = ",",
        line_terminator: str = "\n",
        quote_char: str = '"',
        batch_size: int = 1024,
        datetime_format: str | None = None,
        date_format: str | None = None,
        time_format: str | None = None,
        float_scientific: bool | None = None,
        float_precision: int | None = None,
        null_value: str | None = None,
        quote_style: CsvQuoteStyle | None = None,
    ) -> Self:
        return self.pipe(
            Write(
                filename,
                "csv",
                include_bom=include_bom,
                include_header=include_header,
                separator=separator,
                line_terminator=line_terminator,
                quote_char=quote_char,
                batch_size=batch_size,
                datetime_format=datetime_format,
                date_format=date_format,
                time_format=time_format,
                float_scientific=float_scientific,
                float_precision=float_precision,
                null_value=null_value,
                quote_style=quote_style,
            )
        )

    def write_parquet(
        self,
        filename: str,
        *,
        compression: ParquetCompression = "zstd",
        compression_level: int | None = None,
        statistics: bool | str | dict[str, bool] = True,
        row_group_size: int | None = None,
        data_page_size: int | None = None,
        use_pyarrow: bool = False,
        pyarrow_options: dict[str, Any] | None = None,
        partition_by: str | Sequence[str] | None = None,
        partition_chunk_size_bytes: int = 4_294_967_296,
    ) -> Self:
        return self.pipe(
            Write(
                filename,
                "parquet",
                compression=compression,
                compression_level=compression_level,
                statistics=statistics,
                row_group_size=row_group_size,
                data_page_size=data_page_size,
                use_pyarrow=use_pyarrow,
                pyarrow_options=pyarrow_options,
                partition_by=partition_by,
                partition_chunk_size_bytes=partition_chunk_size_bytes,
            )
        )

    def write_avro(
        self,
        filename: str,
        compression: AvroCompression = "uncompressed",
        name: str = "",
    ) -> Self:
        return self.pipe(Write(filename, "avro", compression, name))

    def print(self) -> Self:
        return self.pipe(Print())

    def display(self) -> Self:
        return self.pipe(Display())

    @property
    def stats(self) -> StatNameSpace:
        return StatNameSpace(self)

    @property
    def horizontal(self) -> HorizontalNameSpace:
        return HorizontalNameSpace(self)

    @property
    def split(self) -> SplitNameSpace:
        return SplitNameSpace(self)

    @property
    def plot(self) -> PlotNameSpace:
        return PlotNameSpace(self)

    @property
    def transform(self) -> TransformNameSpace:
        return TransformNameSpace(self)
