import polars as pl
from polars.testing import assert_frame_equal

from polars_ml.pipeline import LazyPipeline
from polars_ml.pipeline.basic import LazyApply


def test_lazy_pipeline_flow() -> None:
    df = pl.DataFrame({"a": [1, 2]})

    pipe = LazyPipeline(
        LazyApply(lambda x: x.with_columns(pl.col("a") + 1)),
        LazyApply(lambda x: x.with_columns(pl.col("a") * 2)),
    )

    transformed = pipe.transform(df.lazy()).collect()
    expected = pl.DataFrame({"a": [4, 6]})
    assert_frame_equal(transformed, expected)

    transformed_ft = pipe.fit_transform(df)
    assert_frame_equal(transformed_ft, expected)


def test_lazy_pipeline_methods_chaining() -> None:
    df = pl.DataFrame({"a": [3, 1, 2], "b": [10, 20, 30]})

    pipe = LazyPipeline().sort("a").with_columns(pl.col("b") * 2)
    transformed = pipe.transform(df.lazy()).collect()

    expected = pl.DataFrame({"a": [1, 2, 3], "b": [40, 60, 20]})
    assert_frame_equal(transformed, expected)


def test_lazy_pipeline_more_data() -> None:
    from typing import Callable, Self

    df = pl.DataFrame({"a": [1, 2]})
    val = pl.DataFrame({"a": [3, 4]})

    class TrackingTransformer(LazyApply):
        def __init__(self, func: Callable[[pl.LazyFrame], pl.LazyFrame]) -> None:
            super().__init__(func)
            self.fitted_more_data: dict[str, pl.DataFrame] = {}

        def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
            self.fitted_more_data = more_data
            return self

    tracker = TrackingTransformer(lambda x: x)
    pipe = LazyPipeline(LazyApply(lambda x: x.with_columns(pl.col("a") + 10)), tracker)

    pipe.fit(df, val=val)

    assert "val" in tracker.fitted_more_data
    expected_val = pl.DataFrame({"a": [13, 14]})
    assert_frame_equal(tracker.fitted_more_data["val"], expected_val)
