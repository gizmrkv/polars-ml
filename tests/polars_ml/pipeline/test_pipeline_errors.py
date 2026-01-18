import polars as pl
import pytest

from polars_ml.base import Transformer
from polars_ml.pipeline.pipeline import Pipeline


class FailingTransformer(Transformer):
    def fit(self, data, **more_data):
        raise ValueError("Intentional failure in fit")

    def transform(self, data):
        raise ValueError("Intentional failure in transform")

    def fit_transform(self, data, **more_data):
        raise ValueError("Intentional failure in fit_transform")


def test_pipeline_fit_error_context() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    pipeline = Pipeline(
        FailingTransformer(),  # Step 0
    )

    with pytest.raises(ValueError) as excinfo:
        pipeline.fit(df)

    assert "Step 0 (FailingTransformer)" in str(excinfo.value)
    assert "Intentional failure in fit" in str(excinfo.value)


def test_pipeline_transform_error_context() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    pipeline = Pipeline(
        FailingTransformer(),  # Step 0
    )

    with pytest.raises(ValueError) as excinfo:
        pipeline.transform(df)

    assert "Step 0 (FailingTransformer)" in str(excinfo.value)
    assert "Intentional failure in transform" in str(excinfo.value)


def test_pipeline_multiple_steps_error_context() -> None:
    class DummyTransformer(Transformer):
        def fit(self, data, **more_data):
            return self

        def transform(self, data):
            return data

        def fit_transform(self, data, **more_data):
            return data

    df = pl.DataFrame({"a": [1, 2, 3]})
    pipeline = Pipeline(
        DummyTransformer(),  # Step 0
        DummyTransformer(),  # Step 1
        FailingTransformer(),  # Step 2
    )

    with pytest.raises(ValueError) as excinfo:
        pipeline.fit(df)

    assert "Step 2 (FailingTransformer)" in str(excinfo.value)
