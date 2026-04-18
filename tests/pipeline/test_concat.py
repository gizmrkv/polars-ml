import polars as pl
from polars.testing import assert_frame_equal

from polars_ml.pipeline.basic import Apply, LazyApply
from polars_ml.pipeline.concat import Concat, LazyConcat


def test_concat_horizontal() -> None:
    df = pl.DataFrame({"a": [1, 2]})

    t1 = Apply(lambda x: x.select(pl.col("a") * 2).rename({"a": "b"}))
    t2 = Apply(lambda x: x.select(pl.col("a") * 3).rename({"a": "c"}))


    concat = Concat([t1, t2], how="horizontal")
    transformed = concat.transform(df)

    expected = pl.DataFrame({"b": [2, 4], "c": [3, 6]})
    assert_frame_equal(transformed, expected)


def test_concat_vertical() -> None:
    df = pl.DataFrame({"a": [1, 2]})

    t1 = Apply(lambda x: x.select(pl.col("a") + 10))
    t2 = Apply(lambda x: x.select(pl.col("a") + 20))

    concat = Concat([t1, t2], how="vertical")
    transformed = concat.transform(df)

    expected = pl.DataFrame({"a": [11, 12, 21, 22]})
    assert_frame_equal(transformed, expected)


def test_lazy_concat_horizontal() -> None:
    df = pl.DataFrame({"a": [1, 2]})

    t1 = LazyApply(lambda x: x.select(pl.col("a") * 2).rename({"a": "b"}))
    t2 = LazyApply(lambda x: x.select(pl.col("a") * 3).rename({"a": "c"}))

    concat = LazyConcat([t1, t2], how="horizontal")
    transformed = concat.transform(df.lazy()).collect()

    expected = pl.DataFrame({"b": [2, 4], "c": [3, 6]})
    assert_frame_equal(transformed, expected)


def test_lazy_concat_vertical() -> None:
    df = pl.DataFrame({"a": [1, 2]})

    t1 = LazyApply(lambda x: x.select(pl.col("a") + 10))
    t2 = LazyApply(lambda x: x.select(pl.col("a") + 20))

    concat = LazyConcat([t1, t2], how="vertical")
    transformed = concat.transform(df.lazy()).collect()

    expected = pl.DataFrame({"a": [11, 12, 21, 22]})
    assert_frame_equal(transformed, expected)


def test_concat_fit() -> None:
    from typing import Callable, Self

    df = pl.DataFrame({"a": [1, 2]})

    class MockTransformer(Apply):
        def __init__(self, func: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
            super().__init__(func)
            self.is_fitted = False

        def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
            self.is_fitted = True
            return self

    t1 = MockTransformer(lambda x: x)
    t2 = MockTransformer(lambda x: x)

    concat = Concat([t1, t2])
    concat.fit(df)

    assert t1.is_fitted
    assert t2.is_fitted
