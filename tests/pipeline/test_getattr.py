import polars as pl
from polars.testing import assert_frame_equal

from polars_ml.pipeline.getattr import GetAttr, LazyGetAttr


def test_get_attr_basic() -> None:
    df = pl.DataFrame({"a": [3, 1, 2]})

    # Test sort method via GetAttr
    getattr_step = GetAttr("sort", None, "a")
    transformed = getattr_step.transform(df)

    expected = pl.DataFrame({"a": [1, 2, 3]})
    assert_frame_equal(transformed, expected)


def test_get_attr_with_object() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    external_df = pl.DataFrame({"b": [3, 4]})

    # GetAttr can call methods on a provided object
    # Here we simulate a join by calling join on external_df
    # Note: GetAttr(attr, obj, *args, **kwargs) -> obj.attr(*args, **kwargs)
    # external_df.join(df, ...) -> cross join
    getattr_step = GetAttr("join", external_df, df, how="cross")
    transformed = getattr_step.transform(df)

    # external_df.join(df, ...) -> cross join
    expected = external_df.join(df, how="cross")
    assert_frame_equal(transformed, expected)


def test_lazy_get_attr_basic() -> None:
    df = pl.DataFrame({"a": [3, 1, 2]})

    getattr_step = LazyGetAttr("sort", None, "a")
    transformed = getattr_step.transform(df.lazy()).collect()

    expected = pl.DataFrame({"a": [1, 2, 3]})
    assert_frame_equal(transformed, expected)


def test_get_attr_fit_transform_nested() -> None:
    # Test that GetAttr correctly fits and transforms its arguments if they are Transformers
    from typing import Callable, Self

    from polars_ml.pipeline.basic import Apply

    df = pl.DataFrame({"a": [1, 2]})

    class MockTransformer(Apply):
        def __init__(self, func: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
            super().__init__(func)
            self.is_fitted = False

        def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
            self.is_fitted = True
            return self

        def fit_transform(
            self, data: pl.DataFrame, **more_data: pl.DataFrame
        ) -> pl.DataFrame:
            self.is_fitted = True
            return self.transform(data)

    t_arg = MockTransformer(lambda x: x.select(pl.col("a") * 10))

    # GetAttr("join", None, t_arg, ...) -> data.join(t_arg.fit_transform(data), ...)
    # This is a bit complex but tests the nested Transformer logic in GetAttr
    getattr_step = GetAttr("join", None, t_arg, how="cross")

    transformed = getattr_step.fit_transform(df)

    assert t_arg.is_fitted
    expected = df.join(df.select(pl.col("a") * 10), how="cross")
    assert_frame_equal(transformed, expected)
