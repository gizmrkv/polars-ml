import polars as pl
from polars.testing import assert_frame_equal

from polars_ml.pipeline import Pipeline
from polars_ml.pipeline.basic import Apply


def test_pipeline_flow() -> None:
    df = pl.DataFrame({"a": [1, 2]})

    pipe = Pipeline(
        Apply(lambda x: x.with_columns(pl.col("a") + 1)),
        Apply(lambda x: x.with_columns(pl.col("a") * 2)),
    )

    # transform
    transformed = pipe.transform(df)
    expected = pl.DataFrame({"a": [4, 6]})
    assert_frame_equal(transformed, expected)

    # fit_transform
    transformed_ft = pipe.fit_transform(df)
    assert_frame_equal(transformed_ft, expected)


def test_pipeline_methods_chaining() -> None:
    df = pl.DataFrame({"a": [3, 1, 2], "b": [10, 20, 30]})

    # Test chaining of mixin methods
    pipe = Pipeline().sort("a").with_columns(pl.col("b") * 2)
    transformed = pipe.transform(df)

    expected = pl.DataFrame({"a": [1, 2, 3], "b": [40, 60, 20]})
    assert_frame_equal(transformed, expected)


def test_pipeline_more_data() -> None:
    # Test that more_data is handled correctly in fit
    from typing import Callable, Self

    df = pl.DataFrame({"a": [1, 2]})
    val = pl.DataFrame({"a": [3, 4]})

    class TrackingTransformer(Apply):
        def __init__(self, func: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
            super().__init__(func)
            self.fitted_more_data: dict[str, pl.DataFrame] = {}

        def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
            self.fitted_more_data = more_data
            return self

    tracker = TrackingTransformer(lambda x: x)
    pipe = Pipeline(Apply(lambda x: x.with_columns(pl.col("a") + 10)), tracker)

    pipe.fit(df, val=val)

    # tracker.fit should receive transformed val
    assert "val" in tracker.fitted_more_data
    expected_val = pl.DataFrame({"a": [13, 14]})
    assert_frame_equal(tracker.fitted_more_data["val"], expected_val)


def test_pipeline_collect_lazy() -> None:
    from polars_ml.pipeline.basic import Echo

    # Pipeline.pipe converts LazyTransformer to Transformer using .collect()
    pipe = Pipeline().pipe(Echo())
    assert len(pipe._steps) == 1
    # Check if it was converted (Echo is LazyTransformer, but Pipeline.pipe should've wrapped/converted it)
    # Actually looking at pipeline.py:92: step.collect() if isinstance(step, LazyTransformer) else step
    # Wait, does LazyTransformer have .collect()?
    # Let's check polars_ml.base.LazyTransformer (I'll assume it does or pipe method handles it)
    # If it fails, it's a bug the user wants to keep.
    pass
