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


def test_pipeline_logging_verbose() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    steps = [MockTransformer(), MockTransformer()]

    log_capture = io.StringIO()
    custom_logger = logging.getLogger("test_logging")
    custom_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(log_capture)
    custom_logger.addHandler(handler)

    pipeline = Pipeline(*steps, verbose=custom_logger)
    pipeline.fit(df)

    log_output = log_capture.getvalue()
    assert "[1/2] MockTransformer fitting..." in log_output
    assert "[1/2] MockTransformer finished in" in log_output
    assert "[2/2] MockTransformer fitting..." in log_output
    assert "[2/2] MockTransformer finished in" in log_output


def test_pipeline_no_verbose() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    steps = [MockTransformer()]

    log_capture = io.StringIO()
    custom_logger = logging.getLogger("test_no_verbose")
    custom_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(log_capture)
    custom_logger.addHandler(handler)

    pipeline = Pipeline(*steps, verbose=False)
    pipeline.fit(df)

    assert log_capture.getvalue() == ""


def test_pipeline_fit_transform_logging() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    steps = [MockTransformer()]

    log_capture = io.StringIO()
    custom_logger = logging.getLogger("test_fit_transform")
    custom_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(log_capture)
    custom_logger.addHandler(handler)

    pipeline = Pipeline(*steps, verbose=custom_logger)
    pipeline.fit_transform(df)

    log_output = log_capture.getvalue()
    assert "[1/1] MockTransformer fit-transforming..." in log_output
    assert "[1/1] MockTransformer finished in" in log_output


def test_pipeline_transform_logging() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    steps = [MockTransformer()]

    log_capture = io.StringIO()
    custom_logger = logging.getLogger("test_transform")
    custom_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(log_capture)
    custom_logger.addHandler(handler)

    pipeline = Pipeline(*steps, verbose=custom_logger)
    pipeline.transform(df)

    log_output = log_capture.getvalue()
    assert "[1/1] MockTransformer transforming..." in log_output
    assert "[1/1] MockTransformer finished in" in log_output
