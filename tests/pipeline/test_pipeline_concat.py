import polars as pl
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.base import Transformer
from polars_ml.pipeline.pipeline import Pipeline


class AddConst(Transformer):
    def __init__(self, value: int):
        self.value = value

    def transform(self, data: DataFrame) -> DataFrame:
        return data.select(pl.all() + self.value)


def test_concat_basic():
    df = DataFrame({"a": [1, 2]})

    pipeline = Pipeline().concat([AddConst(1), AddConst(10)])

    result = pipeline.fit_transform(df)

    expected = DataFrame({"a": [2, 3, 11, 12]})
    assert_frame_equal(result, expected)


def test_concat_nested_pipeline():
    df = DataFrame({"a": [1]})

    p1 = Pipeline(AddConst(1))
    p2 = Pipeline(AddConst(10))

    main_pipe = Pipeline().concat([p1, p2])

    result = main_pipe.fit_transform(df)
    expected = DataFrame({"a": [2, 11]})

    assert_frame_equal(result, expected)
