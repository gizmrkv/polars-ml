import polars as pl
import pytest

from polars_ml.exceptions import NotFittedError
from polars_ml.pipeline.combine import Combine


def test_combine_basic() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [True, False]})

    combine = Combine(["a", "b"], n=2)
    combine.fit(df)
    transformed = combine.transform(df.lazy()).collect()

    assert "a_b" in transformed.columns
    assert transformed["a_b"].dtype == pl.Struct

    expected_structs = pl.Series("a_b", [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    assert (transformed["a_b"] == expected_structs).all()


def test_combine_delimiter() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    combine = Combine(["a", "b"], n=2, delimiter="-")
    combine.fit(df)
    transformed = combine.transform(df.lazy()).collect()

    assert "a-b" in transformed.columns


def test_combine_not_fitted() -> None:
    combine = Combine(["a", "b"], n=2)
    with pytest.raises(NotFittedError):
        combine.transform(pl.LazyFrame({"a": [1], "b": ["x"]}))


def test_combine_permutations_instead_of_combinations() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})

    combine = Combine(["a", "b", "c"], n=2)
    combine.fit(df)
    transformed = combine.transform(df.lazy()).collect()

    assert "a_b" in transformed.columns
    assert "a_c" in transformed.columns
    assert "b_c" in transformed.columns


def test_pipeline_combine() -> None:
    from polars_ml.pipeline import Pipeline

    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    pipe = Pipeline().combine(["a", "b"], n=2)
    transformed = pipe.fit_transform(df)
    assert "a_b" in transformed.columns


def test_lazy_pipeline_combine() -> None:
    from polars_ml.pipeline import LazyPipeline

    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    pipe = LazyPipeline().combine(["a", "b"], n=2)
    transformed = pipe.fit_transform(df)
    assert "a_b" in transformed.columns
