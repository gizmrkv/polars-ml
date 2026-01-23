import numpy as np
import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal, assert_series_equal

from polars_ml.exceptions import NotFittedError
from polars_ml.pipeline.pipeline import Pipeline


def test_standard_scale_basic():
    df = DataFrame({"val": [1.0, 2.0, 3.0]})

    pipeline = Pipeline().standard_scale("val")
    result = pipeline.fit_transform(df)

    encoded = result["val"]
    assert encoded.dtype == pl.Float64
    assert_series_equal(encoded, pl.Series("val", [-1.0, 0.0, 1.0]))


def test_min_max_scale_basic():
    df = DataFrame({"val": [1.0, 2.0, 3.0]})

    pipeline = Pipeline().min_max_scale("val")
    result = pipeline.fit_transform(df)

    encoded = result["val"]
    assert encoded.dtype == pl.Float64
    assert_series_equal(encoded, pl.Series("val", [0.0, 0.5, 1.0]))


def test_robust_scale_basic():
    df = DataFrame({"val": [1.0, 2.0, 3.0, 100.0]})

    pipeline = Pipeline().robust_scale("val")
    result = pipeline.fit_transform(df)

    assert "val" in result.columns
    assert result["val"].dtype == pl.Float64


def test_scale_grouped():
    df = DataFrame({"group": ["a", "a", "b", "b"], "val": [10.0, 20.0, 100.0, 200.0]})

    pipeline = Pipeline().standard_scale("val", by="group")
    result = pipeline.fit_transform(df)

    params_a = result.filter(pl.col("group") == "a")["val"].to_list()
    params_b = result.filter(pl.col("group") == "b")["val"].to_list()

    np.testing.assert_allclose(params_a, params_b)


def test_scale_inverse_context():
    df = DataFrame({"val": [1.0, 5.0, 10.0]})

    pipe = Pipeline()
    with pipe.min_max_scale("val", inverse_mapping={"val_restored": "val"}):
        pass

    result = pipe.fit_transform(df)

    assert "val" in result.columns
    assert "val_restored" in result.columns

    original_vals = df["val"]
    restored_vals = result["val_restored"].rename("val")

    assert_series_equal(original_vals, restored_vals)


def test_scale_not_fitted():
    df = DataFrame({"val": [1.0]})
    pipeline = Pipeline().standard_scale("val")
    with pytest.raises(NotFittedError):
        pipeline.transform(df)
