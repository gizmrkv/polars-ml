import polars as pl
import pytest

from polars_ml.exceptions import NotFittedError
from polars_ml.pipeline.power import BoxCoxTransform, YeoJohnsonTransform


def test_boxcox_basic() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})

    transformer = BoxCoxTransform("a")
    transformer.fit(df)
    transformed = transformer.transform(df.lazy()).collect()

    assert transformed["a"].dtype == pl.Float64
    assert transformed["a"].n_unique() == 5
    assert pytest.approx(transformed["a"][0]) == 0.0


def test_boxcox_by_group() -> None:
    df = pl.DataFrame(
        {"g": ["a", "a", "a", "b", "b", "b"], "v": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]}
    )

    transformer = BoxCoxTransform("v", by="g")
    transformer.fit(df)
    transformer.transform(df.lazy()).collect()

    assert transformer.maxlog.height == 2
    assert "v_maxlog" in transformer.maxlog.columns


def test_yeojohnson_basic() -> None:
    df = pl.DataFrame({"a": [-2.0, -1.0, 0.0, 1.0, 2.0]})

    transformer = YeoJohnsonTransform("a")
    transformer.fit(df)
    transformed = transformer.transform(df.lazy()).collect()

    assert transformed["a"].dtype == pl.Float64
    assert transformed["a"].n_unique() == 5


def test_power_not_fitted() -> None:
    transformer = BoxCoxTransform("a")
    with pytest.raises(NotFittedError):
        transformer.transform(pl.LazyFrame({"a": [1.0]}))


def test_pipeline_boxcox() -> None:
    from polars_ml.pipeline import Pipeline

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    pipe = Pipeline().boxcox("a")
    transformed = pipe.fit_transform(df)
    assert transformed["a"].dtype == pl.Float64
    assert pytest.approx(transformed["a"][0]) == 0.0


def test_lazy_pipeline_boxcox() -> None:
    from polars_ml.pipeline import LazyPipeline

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    pipe = LazyPipeline().boxcox("a")
    transformed = pipe.fit_transform(df)
    assert transformed["a"].dtype == pl.Float64
    assert pytest.approx(transformed["a"][0]) == 0.0


def test_pipeline_yeojohnson() -> None:
    from polars_ml.pipeline import Pipeline

    df = pl.DataFrame({"a": [-2.0, -1.0, 0.0, 1.0, 2.0]})
    pipe = Pipeline().yeojohnson("a")
    transformed = pipe.fit_transform(df)
    assert transformed["a"].dtype == pl.Float64
