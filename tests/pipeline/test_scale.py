import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_ml.exceptions import NotFittedError
from polars_ml.pipeline.scale import MinMaxScale, RobustScale, StandardScale


def test_standard_scale_basic() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})

    scaler = StandardScale("a")
    scaler.fit(df)
    transformed = scaler.transform(df.lazy()).collect()

    expected = pl.DataFrame({"a": [-1.0, 0.0, 1.0]})
    assert_frame_equal(transformed, expected)


def test_min_max_scale_basic() -> None:
    df = pl.DataFrame({"a": [1.0, 3.0, 5.0]})

    scaler = MinMaxScale("a")
    scaler.fit(df)
    transformed = scaler.transform(df.lazy()).collect()

    expected = pl.DataFrame({"a": [0.0, 0.5, 1.0]})
    assert_frame_equal(transformed, expected)


def test_robust_scale_basic() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})

    scaler = RobustScale("a", quantile_range=(0.25, 0.75))
    scaler.fit(df)
    transformed = scaler.transform(df.lazy()).collect()

    expected = pl.DataFrame({"a": [-1.0, -0.5, 0.0, 0.5, 1.0]})
    assert_frame_equal(transformed, expected)


def test_scale_by_group() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, 3.0, 10.0, 30.0]})

    scaler = MinMaxScale("v", by="g")
    scaler.fit(df)
    transformed = scaler.transform(df.lazy()).collect()

    expected = pl.DataFrame({"g": ["a", "a", "b", "b"], "v": [0.0, 1.0, 0.0, 1.0]})
    assert_frame_equal(transformed, expected)


def test_scale_not_fitted() -> None:
    scaler = StandardScale("a")
    with pytest.raises(NotFittedError):
        scaler.transform(pl.LazyFrame({"a": [1.0]}))


def test_pipeline_standard_scale() -> None:
    from polars_ml.pipeline import Pipeline

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    pipe = Pipeline().standard_scale("a")
    transformed = pipe.fit_transform(df)
    expected = pl.DataFrame({"a": [-1.0, 0.0, 1.0]})
    assert_frame_equal(transformed, expected)


def test_lazy_pipeline_standard_scale() -> None:
    from polars_ml.pipeline import LazyPipeline

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    pipe = LazyPipeline().standard_scale("a")
    transformed = pipe.fit_transform(df)
    expected = pl.DataFrame({"a": [-1.0, 0.0, 1.0]})
    assert_frame_equal(transformed, expected)


def test_pipeline_min_max_scale() -> None:
    from polars_ml.pipeline import Pipeline

    df = pl.DataFrame({"a": [1.0, 3.0, 5.0]})
    pipe = Pipeline().min_max_scale("a")
    transformed = pipe.fit_transform(df)
    expected = pl.DataFrame({"a": [0.0, 0.5, 1.0]})
    assert_frame_equal(transformed, expected)


def test_pipeline_robust_scale() -> None:
    from polars_ml.pipeline import Pipeline

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    pipe = Pipeline().robust_scale("a", quantile_range=(0.25, 0.75))
    transformed = pipe.fit_transform(df)
    expected = pl.DataFrame({"a": [-1.0, -0.5, 0.0, 0.5, 1.0]})
    assert_frame_equal(transformed, expected)
