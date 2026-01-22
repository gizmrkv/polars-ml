import numpy as np
import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal, assert_series_equal

from polars_ml.exceptions import NotFittedError
from polars_ml.pipeline.pipeline import Pipeline


def test_boxcox_basic():
    df = DataFrame({"val": [1.0, 2.0, 3.0]})
    pipeline = Pipeline().boxcox("val")
    result = pipeline.fit_transform(df)

    assert "val" in result.columns
    assert result["val"].dtype == pl.Float64
    vals = result["val"].to_list()
    assert vals[0] < vals[1] < vals[2]


def test_yeojohnson_basic():
    df = DataFrame({"val": [-10.0, 0.0, 10.0]})
    pipeline = Pipeline().yeojohnson("val")
    result = pipeline.fit_transform(df)

    assert "val" in result.columns
    assert result["val"].dtype == pl.Float64
    vals = result["val"].to_list()
    assert vals[0] < vals[1] < vals[2]


def test_boxcox_grouped():
    df = DataFrame({"group": ["a", "a", "b", "b"], "val": [1.0, 2.0, 100.0, 200.0]})

    pipeline = Pipeline().boxcox("val", by="group")
    result = pipeline.fit_transform(df)

    assert result.shape == df.shape
    assert result["val"].dtype == pl.Float64


def test_power_inverse_context():
    df = DataFrame({"val": [1.0, 5.0, 10.0]})

    pipe = Pipeline()

    with pipe.boxcox("val", inverse_mapping={"val_restored": "val"}):
        pass

    result = pipe.fit_transform(df)

    assert "val" in result.columns
    assert "val_restored" in result.columns

    original_vals = df["val"]
    restored_vals = result["val_restored"].rename("val")

    assert_series_equal(original_vals, restored_vals)


def test_power_not_fitted():
    df = DataFrame({"val": [1.0, 2.0]})
    pipeline = Pipeline().boxcox("val")
    with pytest.raises(NotFittedError):
        pipeline.transform(df)
