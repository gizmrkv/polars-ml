from typing import Any

import polars as pl
import pytest
from polars import DataFrame, Series
from polars.testing import assert_frame_equal

from polars_ml import Pipeline


@pytest.fixture
def test_data_basic():
    return DataFrame(
        {
            "f0": [1, 2, 3, 4, 5],
            "f1": [1, None, 3, 4, None],
            "f2": [0.1, 0.2, float("nan"), 0.4, 0.5],
            "f3": ["a", "a", "b", "b", "c"],
        }
    )


def assert_frame_getattr_equal(data: DataFrame, method: str, *args: Any, **kwargs: Any):
    pp: Pipeline = getattr(Pipeline(), method)(*args, **kwargs)
    out = pp.transform(data.clone())
    exp = getattr(data.clone(), method)(*args, **kwargs)
    assert_frame_equal(out, exp)


def test_pipeline_bottom_k_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "bottom_k", 2, by="f0")


def test_pipeline_cast_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "cast", dtypes={"f0": pl.String})


def test_pipeline_clear_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "clear")


def test_pipeline_clone_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "clone")


def test_pipeline_count_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "count")


def test_pipeline_describe_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "describe")


def test_pipeline_drop_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "drop", "f0")


def test_pipeline_drop_nans_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "drop_nans", "f2")


def test_pipeline_drop_nulls_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "drop_nulls", "f1")


def test_pipeline_extend_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "extend", test_data_basic)


def test_pipeline_fill_nan_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "fill_nan", 0.0)


def test_pipeline_fill_null_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "fill_null", "NULL")


def test_pipeline_filter_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "filter", pl.col("f0") > 2)


def test_pipeline_gather_every_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "gather_every", 2)


def test_pipeline_head_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "head", 2)


def test_pipeline_insert_column_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(
        test_data_basic,
        "insert_column",
        1,
        Series("f0.5", range(test_data_basic.height)),
    )


def test_pipeline_interpolate_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "interpolate")


def test_pipeline_join_basic(test_data_basic: DataFrame):
    other = DataFrame(
        {
            "f3": ["a", "b", "c"],
            "j0": [1, 2, 3],
        }
    )
    assert_frame_getattr_equal(test_data_basic, "join", other, on="f3")
    assert_frame_getattr_equal(test_data_basic, "join_asof", other, on="f3")


def test_pipeline_limit_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "limit", 2)


def test_pipeline_mean_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "mean")


def test_pipeline_median_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "median")


def test_pipeline_min_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "min")


def test_pipeline_null_count_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "null_count")


def test_pipeline_pivot_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "pivot", "f3", values="f0")


def test_pipeline_product_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "product")


def test_pipeline_quantile_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "quantile", 0.5)


def test_pipeline_rechunk_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "rechunk")


def test_pipeline_rename_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "rename", {"f0": "f0_new"})


def test_pipeline_replace_column_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(
        test_data_basic,
        "replace_column",
        1,
        Series("f0.5", range(test_data_basic.height)),
    )


def test_pipeline_sample_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "sample", 2, seed=42)


def test_pipeline_select_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "select", "f0")


def test_pipeline_select_seq_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "select_seq", "f0")


def test_pipeline_set_sorted_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "set_sorted", "f0")


def test_pipeline_shift_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "shift", 1)


def test_pipeline_shrink_to_fit_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "shrink_to_fit")


def test_pipeline_slice_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "slice", 1, 3)


def test_pipeline_sort_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "sort", pl.all())


def test_pipeline_sql_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "sql", "SELECT * FROM self")


def test_pipeline_std_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "std")


def test_pipeline_sum_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "sum")


def test_pipeline_tail_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "tail", 2)


def test_pipeline_top_k_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "top_k", 2, by="f0")


def test_pipeline_transpose_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "transpose")


def test_pipeline_unique_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "unique", maintain_order=True)


def test_pipeline_unpivot_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "unpivot", "f3")


def test_pipeline_var_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "var")


def test_pipeline_vstack_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "vstack", test_data_basic)


def test_pipeline_with_columns_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "with_columns", pl.col("f0") * 2)


def test_pipeline_with_columns_seq_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "with_columns_seq", pl.col("f0") * 2)


def test_pipeline_with_row_index_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "with_row_index")


def test_pipeline_to_dummies_basic(test_data_basic: DataFrame):
    assert_frame_getattr_equal(test_data_basic, "to_dummies", "f3")


def assert_frame_do_nothing(data: DataFrame, method: str, *args: Any, **kwargs: Any):
    pp: Pipeline = getattr(Pipeline(), method)(*args, **kwargs)
    out = pp.transform(data)
    assert_frame_equal(out, data)


def test_pipeline_print_basic(test_data_basic: DataFrame):
    assert_frame_do_nothing(test_data_basic, "print")


def test_pipeline_display_basic(test_data_basic: DataFrame):
    assert_frame_do_nothing(test_data_basic, "display")


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
