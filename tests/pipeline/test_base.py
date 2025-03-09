from copy import deepcopy

import polars as pl
import pytest
from polars import DataFrame, Series
from polars.testing import assert_frame_equal

from polars_ml import Pipeline


@pytest.mark.parametrize(
    "data",
    [
        DataFrame(
            {
                "f0": [1, 2, 3, 4, 5],
                "f1": [1, None, 3, 4, None],
                "f2": [0.1, 0.2, float("nan"), 0.4, 0.5],
                "f3": ["a", "a", "b", "b", "c"],
            }
        )
    ],
)
def test_pipeline_getattr(data: DataFrame):
    join_other = DataFrame(
        {
            "f3": ["a", "b", "c"],
            "j0": [1, 2, 3],
        }
    )
    for method, args, kwargs in [
        ("bottom_k", [2], {"by": "f0"}),
        ("cast", [], {"dtypes": {"f0": pl.String}}),
        ("clear", [], {}),
        ("clone", [], {}),
        ("count", [], {}),
        ("describe", [], {}),
        ("drop", ["f0"], {}),
        ("drop_nans", ["f2"], {}),
        ("drop_nulls", ["f1"], {}),
        ("extend", [data], {}),
        ("fill_nan", [0.0], {}),
        ("fill_null", ["missing"], {}),
        ("filter", [pl.col("f0") > 2], {}),
        ("gather_every", [2], {}),
        ("head", [2], {}),
        ("insert_column", [1, Series("f0.5", range(data.height))], {}),
        ("interpolate", [], {}),
        ("join", [join_other.clone()], {"on": "f3"}),
        ("join_asof", [join_other.clone()], {"on": "f3"}),
        ("limit", [2], {}),
        ("mean", [], {}),
        ("median", [], {}),
        ("min", [], {}),
        ("null_count", [], {}),
        ("pivot", ["f3"], {"values": "f0"}),
        ("product", [], {}),
        ("quantile", [0.5], {}),
        ("rechunk", [], {}),
        ("rename", [{c: c + "_new" for c in data.columns}], {}),
        ("replace_column", [1, Series("f0.5", range(data.height))], {}),
        ("sample", [2], {"seed": 42}),
        ("select", ["f0"], {}),
        ("select_seq", ["f0"], {}),
        ("set_sorted", ["f0"], {}),
        ("shift", [1], {}),
        ("shrink_to_fit", [], {}),
        ("slice", [1, 3], {}),
        ("sort", [pl.all()], {}),
        ("sql", ["SELECT * FROM self"], {}),
        ("std", [], {}),
        ("sum", [], {}),
        ("tail", [2], {}),
        ("top_k", [2], {"by": "f0"}),
        ("transpose", [], {}),
        ("unique", [], {"maintain_order": True}),
        ("unpivot", ["f3"], {}),
        ("var", [], {}),
        ("vstack", [data.clone()], {}),
        ("with_columns", [pl.col("f0") * 2], {}),
        ("with_columns_seq", [pl.col("f0") * 2], {}),
        ("with_row_index", [], {}),
        ("to_dummies", ["f3"], {}),
    ]:
        pp: Pipeline = getattr(Pipeline(), method)(*args, **kwargs)
        out = pp.transform(deepcopy(data))
        exp = getattr(deepcopy(data), method)(*args, **kwargs)
        assert_frame_equal(out, exp)


def test_pipeline_print():
    data = DataFrame({"f0": [1, 2, 3, 4, 5]})
    pp = Pipeline().print()
    out = pp.transform(data)
    assert_frame_equal(out, data)


def test_pipeline_display():
    data = DataFrame({"f0": [1, 2, 3, 4, 5]})
    pp = Pipeline().display()
    out = pp.transform(data)
    assert_frame_equal(out, data)


def test_pipeline_sort_columns():
    data = DataFrame({"b": 0, "c": True, "a": 0.1})

    pp = Pipeline().sort_columns()
    out = pp.transform(data)
    assert_frame_equal(out, data.select("c", "a", "b"))

    pp = Pipeline().sort_columns("name")
    out = pp.transform(data)
    assert_frame_equal(out, data.select("a", "b", "c"))


def test_pipeline_group_by_then():
    train_data = DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    valid_data = DataFrame({"a": [1, 2, 3]})

    pp = Pipeline().group_by_then("a", pl.sum("b"), maintain_order=True)
    pp.fit(train_data)
    out = pp.transform(valid_data)
    exp = DataFrame({"a": [1, 2, 3], "b": [3, 7, None]})
    assert_frame_equal(out, exp)


@pytest.mark.parametrize(
    ["data", "expected"],
    [
        (
            DataFrame(
                {"f0": [1, 1, 2, 2, 2], "f1": [0.1, None, 0.2, None, 0.4]}
            ).with_row_index(),
            DataFrame(
                {"f0": [1, 1, 2, 2, 2], "f1": [0.1, 0.1, 0.2, 0.3, 0.4]}
            ).with_row_index(),
        ),
    ],
)
def test_pipeline_impute(data: DataFrame, expected: DataFrame):
    print("#" * 80)
    pp = Pipeline().impute(
        Pipeline()
        .group_by_then("f0", pl.mean("f1"), maintain_order=True)
        .print()
        .select("f1"),
        "f1",
        maintain_order=True,
    )
    out = pp.fit_transform(data)
    assert_frame_equal(out, expected)
