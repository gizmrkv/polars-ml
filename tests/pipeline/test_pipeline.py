from typing import Any

import polars as pl
from polars import DataFrame, Series

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def assert_pipeline_getattr_equal(
    data: DataFrame,
    method: str,
    *args: Any,
    is_inplace: bool = False,
    **kwargs: Any,
):
    assert_component_valid(
        getattr(Pipeline(), method)(*args, **kwargs),
        data.clone(),
        getattr(data.clone(), method)(*args, **kwargs),
        is_inplace=is_inplace,
    )


def test_pipeline_bottom_k_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "bottom_k", 2, by="f0")


def test_pipeline_cast_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "cast", dtypes={"f0": pl.String})


def test_pipeline_clear_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "clear")


def test_pipeline_clone_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "clone")


def test_pipeline_count_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "count")


def test_pipeline_describe_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "describe")


def test_pipeline_drop_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "drop", "f0")


def test_pipeline_drop_nans_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "drop_nans", "f2")


def test_pipeline_drop_nulls_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "drop_nulls", "f1")


def test_pipeline_extend_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(
        test_data_small, "extend", test_data_small, is_inplace=True
    )


def test_pipeline_fill_nan_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "fill_nan", 0.0)


def test_pipeline_fill_null_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "fill_null", "NULL")


def test_pipeline_filter_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "filter", pl.col("f0") > 2)


def test_pipeline_gather_every_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "gather_every", 2)


def test_pipeline_head_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "head", 2)


def test_pipeline_insert_column_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(
        test_data_small,
        "insert_column",
        1,
        Series("f0.5", range(test_data_small.height)),
        is_inplace=True,
    )


def test_pipeline_interpolate_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "interpolate")


def test_pipeline_join_small(test_data_small: DataFrame):
    other = DataFrame(
        {
            "s0": ["a", "b", "c"],
            "other": [1, 2, 3],
        }
    )
    assert_pipeline_getattr_equal(test_data_small, "join", other, on="s0")
    assert_pipeline_getattr_equal(test_data_small, "join_asof", other, on="s0")


def test_pipeline_limit_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "limit", 2)


def test_pipeline_mean_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "mean")


def test_pipeline_median_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "median")


def test_pipeline_min_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "min")


def test_pipeline_null_count_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "null_count")


def test_pipeline_pivot_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "pivot", "s0", values="f0")


def test_pipeline_product_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "product")


def test_pipeline_quantile_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "quantile", 0.5)


def test_pipeline_rechunk_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "rechunk")


def test_pipeline_rename_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "rename", {"f0": "f0_new"})


def test_pipeline_replace_column_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(
        test_data_small,
        "replace_column",
        1,
        Series("f0.5", range(test_data_small.height)),
        is_inplace=True,
    )


def test_pipeline_sample_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "sample", 2, seed=42)


def test_pipeline_select_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "select", "f0")


def test_pipeline_select_seq_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "select_seq", "f0")


def test_pipeline_set_sorted_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "set_sorted", "f0")


def test_pipeline_shift_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "shift", 1)


def test_pipeline_shrink_to_fit_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "shrink_to_fit")


def test_pipeline_slice_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "slice", 1, 3)


def test_pipeline_sort_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "sort", pl.all())


def test_pipeline_sql_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "sql", "SELECT * FROM self")


def test_pipeline_std_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "std")


def test_pipeline_sum_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "sum")


def test_pipeline_tail_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "tail", 2)


def test_pipeline_top_k_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "top_k", 2, by="f0")


def test_pipeline_transpose_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "transpose")


def test_pipeline_unique_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "unique", maintain_order=True)


def test_pipeline_unpivot_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "unpivot", "s0")


def test_pipeline_var_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "var")


def test_pipeline_vstack_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "vstack", test_data_small)


def test_pipeline_with_columns_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "with_columns", pl.col("f0") * 2)


def test_pipeline_with_columns_seq_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "with_columns_seq", pl.col("f0") * 2)


def test_pipeline_with_row_index_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "with_row_index")


def test_pipeline_to_dummies_small(test_data_small: DataFrame):
    assert_pipeline_getattr_equal(test_data_small, "to_dummies", "s0")
