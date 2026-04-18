import polars as pl
from polars.testing import assert_frame_equal

from polars_ml.pipeline.getattr import GetAttr, LazyGetAttr


def test_get_attr_basic() -> None:
    df = pl.DataFrame({"a": [3, 1, 2]})

    getattr_step = GetAttr("sort", None, "a")
    transformed = getattr_step.transform(df)

    expected = pl.DataFrame({"a": [1, 2, 3]})
    assert_frame_equal(transformed, expected)


def test_get_attr_with_object() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    external_df = pl.DataFrame({"b": [3, 4]})

    getattr_step = GetAttr("join", external_df, df, how="cross")
    transformed = getattr_step.transform(df)

    expected = external_df.join(df, how="cross")
    assert_frame_equal(transformed, expected)


def test_lazy_get_attr_basic() -> None:
    df = pl.DataFrame({"a": [3, 1, 2]})

    getattr_step = LazyGetAttr("sort", None, "a")
    transformed = getattr_step.transform(df.lazy()).collect()

    expected = pl.DataFrame({"a": [1, 2, 3]})
    assert_frame_equal(transformed, expected)


def test_get_attr_fit_transform_nested() -> None:
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

    getattr_step = GetAttr("join", None, t_arg, how="cross")

    transformed = getattr_step.fit_transform(df)

    assert t_arg.is_fitted
    expected = df.join(df.select(pl.col("a") * 10), how="cross")
    assert_frame_equal(transformed, expected)
