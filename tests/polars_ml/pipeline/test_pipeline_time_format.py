import io
import logging
from typing import Self

import polars as pl
import pytest
from polars import DataFrame

from polars_ml.base import Transformer
from polars_ml.pipeline import Pipeline


class MockTransformer(Transformer):
    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return data

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        return data


def test_pipeline_time_formatting_seconds():
    pipeline = Pipeline(MockTransformer())
    # < 60s
    assert pipeline._format_time(45.678) == "45.68s"
    assert pipeline._format_time(0.1234) == "0.12s"


def test_pipeline_time_formatting_long_duration():
    pipeline = Pipeline(MockTransformer())
    # > 60s (minutes)
    assert pipeline._format_time(65.432) == "01m 05.43s"
    assert pipeline._format_time(125.0) == "02m 05.00s"
    # > 3600s (hours)
    assert pipeline._format_time(3661.123) == "01h 01m 01.12s"
    assert pipeline._format_time(7200.0) == "02h 00m 00.00s"
    # > 86400s (days)
    assert pipeline._format_time(90061.0) == "1d 01h 01m 01.00s"
    assert pipeline._format_time(172800.0) == "2d 00h 00m 00.00s"


def test_pipeline_logging_with_formatted_time():
    df = pl.DataFrame({"a": [1, 2, 3]})
    steps = [MockTransformer()]

    log_capture = io.StringIO()
    custom_logger = logging.getLogger("test_time")
    custom_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(log_capture)
    custom_logger.addHandler(handler)

    pipeline = Pipeline(*steps, verbose=custom_logger)
    pipeline.fit(df)

    log_output = log_capture.getvalue()
    # Check that it ends with 's' and potentially has 'm' if it took long (though mock is fast)
    assert "finished in" in log_output
    assert log_output.strip().endswith("s")
