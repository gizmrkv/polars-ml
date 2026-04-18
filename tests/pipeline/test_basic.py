import polars as pl
from polars.testing import assert_frame_equal

from polars_ml.pipeline.basic import (
    Apply,
    Const,
    Echo,
    LazyApply,
    LazyConst,
    LazySide,
    Replay,
    Side,
)


def test_echo() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    echo = Echo()
    transformed = echo.transform(df.lazy()).collect()
    assert_frame_equal(transformed, df)


def test_replay() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [3, 4]})
    replay = Replay()
    replay.fit(df1)
    transformed = replay.transform(df2.lazy()).collect()
    assert_frame_equal(transformed, df1)


def test_const() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [3, 4]})
    const = Const(df1)
    transformed = const.transform(df2)
    assert_frame_equal(transformed, df1)


def test_lazy_const() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [3, 4]})
    const = LazyConst(df1.lazy())
    transformed = const.transform(df2.lazy()).collect()
    assert_frame_equal(transformed, df1)


def test_apply() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    apply = Apply(lambda x: x.with_columns(pl.col("a") * 2))
    transformed = apply.transform(df)
    expected = pl.DataFrame({"a": [2, 4]})
    assert_frame_equal(transformed, expected)


def test_lazy_apply() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    apply = LazyApply(lambda x: x.with_columns(pl.col("a") * 2))
    transformed = apply.transform(df.lazy()).collect()
    expected = pl.DataFrame({"a": [2, 4]})
    assert_frame_equal(transformed, expected)


def test_side() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    # Side should execute the transformer but return the original data
    mock_transformer = Apply(lambda x: x.with_columns(pl.col("a") * 2))
    side = Side(mock_transformer)

    transformed = side.transform(df)
    assert_frame_equal(transformed, df)

    # Check fit_transform
    transformed_ft = side.fit_transform(df)
    assert_frame_equal(transformed_ft, df)


def test_lazy_side() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    mock_transformer = LazyApply(lambda x: x.with_columns(pl.col("a") * 2))
    side = LazySide(mock_transformer)

    transformed = side.transform(df.lazy()).collect()
    assert_frame_equal(transformed, df)

    # Check fit_transform
    transformed_ft = side.fit_transform(df)
    assert_frame_equal(transformed_ft, df)


def test_pipeline_echo() -> None:
    from polars_ml.pipeline import Pipeline

    df = pl.DataFrame({"a": [1, 2]})
    pipe = Pipeline().echo()
    transformed = pipe.transform(df)
    assert_frame_equal(transformed, df)


def test_lazy_pipeline_echo() -> None:
    from polars_ml.pipeline import LazyPipeline

    df = pl.DataFrame({"a": [1, 2]})
    pipe = LazyPipeline().echo()
    transformed = pipe.transform(df.lazy()).collect()
    assert_frame_equal(transformed, df)


def test_pipeline_replay() -> None:
    from polars_ml.pipeline import Pipeline

    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [3, 4]})
    pipe = Pipeline().replay()
    pipe.fit(df1)
    transformed = pipe.transform(df2)
    assert_frame_equal(transformed, df1)


def test_lazy_pipeline_replay() -> None:
    from polars_ml.pipeline import LazyPipeline

    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [3, 4]})
    pipe = LazyPipeline().replay()
    pipe.fit(df1)
    transformed = pipe.transform(df2.lazy()).collect()
    assert_frame_equal(transformed, df1)
