import polars as pl
import pytest

from polars_ml.exceptions import NotFittedError
from polars_ml.pipeline.discretize import Discretize


def test_discretize_basic() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})

    # 2 bins (quantiles=2)
    disc = Discretize("a", quantiles=2)
    disc.fit(df)
    transformed = disc.transform(df.lazy()).collect()

    assert "a_disc" in transformed.columns
    # Check that we have 2 levels (categorical/string depending on labels,
    # but by default it's from pl.cut)
    assert transformed["a_disc"].n_unique() == 2


def test_discretize_labels() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})

    disc = Discretize("a", quantiles=2, labels=["low", "high"])
    disc.fit(df)
    transformed = disc.transform(df.lazy()).collect()

    assert transformed["a_disc"].to_list() == ["low"] * 5 + ["high"] * 5


def test_discretize_not_fitted() -> None:
    disc = Discretize("a", quantiles=2)
    with pytest.raises(NotFittedError):
        disc.transform(pl.LazyFrame({"a": [1.0]}))


def test_discretize_multiple_columns() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})

    disc = Discretize("a", "b", quantiles=2, labels=["L", "H"])
    disc.fit(df)
    transformed = disc.transform(df.lazy()).collect()

    assert "a_disc" in transformed.columns
    assert "b_disc" in transformed.columns
    assert transformed["a_disc"].to_list() == ["L", "L", "H", "H"]
    assert transformed["b_disc"].to_list() == ["L", "L", "H", "H"]


def test_pipeline_discretize() -> None:
    from polars_ml.pipeline import Pipeline

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    pipe = Pipeline().discretize("a", quantiles=2, labels=["low", "high"])
    transformed = pipe.fit_transform(df)
    assert transformed["a_disc"].to_list() == ["low"] * 5 + ["high"] * 5


def test_lazy_pipeline_discretize() -> None:
    from polars_ml.pipeline import LazyPipeline

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    pipe = LazyPipeline().discretize("a", quantiles=2, labels=["low", "high"])
    transformed = pipe.fit_transform(df)
    assert transformed["a_disc"].to_list() == ["low"] * 5 + ["high"] * 5
