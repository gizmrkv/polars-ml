import polars as pl
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.pipeline.pipeline import Pipeline


def test_horizontal_sum_mean_max_min():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    p_sum = Pipeline().horizontal.sum(["a", "b", "c"], value_name="h_sum")
    res_sum = p_sum.transform(df)
    expected_sum = df.with_columns(h_sum=pl.Series([12, 15, 18]))
    assert_frame_equal(res_sum, expected_sum)

    p_mean = Pipeline().horizontal.mean(["a", "b", "c"], value_name="h_mean")
    res_mean = p_mean.transform(df)
    expected_mean = df.with_columns(h_mean=pl.Series([4.0, 5.0, 6.0]))
    assert_frame_equal(res_mean, expected_mean)

    p_max = Pipeline().horizontal.max(["a", "b", "c"], value_name="h_max")
    res_max = p_max.transform(df)
    expected_max = df.with_columns(h_max=pl.Series([7, 8, 9]))
    assert_frame_equal(res_max, expected_max)

    p_min = Pipeline().horizontal.min(["a", "b", "c"], value_name="h_min")
    res_min = p_min.transform(df)
    expected_min = df.with_columns(h_min=pl.Series([1, 2, 3]))
    assert_frame_equal(res_min, expected_min)


def test_horizontal_arg_min_max():
    df = DataFrame({"a": [10, 1, 5], "b": [5, 10, 1], "c": [1, 5, 10]})

    p_argmax = Pipeline().horizontal.arg_max(["a", "b", "c"], value_name="h_argmax")
    res_argmax = p_argmax.transform(df)
    assert res_argmax["h_argmax"].to_list() == [["a"], ["b"], ["c"]]

    p_argmin = Pipeline().horizontal.arg_min(["a", "b", "c"], value_name="h_argmin")
    res_argmin = p_argmin.transform(df)
    assert res_argmin["h_argmin"].to_list() == [["c"], ["a"], ["b"]]


def test_horizontal_quantile():
    df = DataFrame({"a": [1, 1], "b": [3, 3], "c": [5, 5]})
    p_quant = Pipeline().horizontal.quantile(
        ["a", "b", "c"], quantile=0.5, value_name="h_median"
    )
    res_quant = p_quant.transform(df)
    expected_quant = df.with_columns(h_median=pl.Series([3.0, 3.0]))
    assert_frame_equal(res_quant, expected_quant)


def test_horizontal_other_aggs():
    df = DataFrame({"a": [1, 1, None], "b": [2, 1, 2], "c": [3, 1, 3]})

    p_count = Pipeline().horizontal.count(["a", "b", "c"], value_name="h_count")
    res_count = p_count.transform(df)
    expected_count = df.with_columns(h_count=pl.Series([3, 3, 2], dtype=pl.UInt32))
    assert_frame_equal(res_count, expected_count)

    p_nunique = Pipeline().horizontal.n_unique(["a", "b", "c"], value_name="h_nunique")
    res_nunique = p_nunique.transform(df)
    expected_nunique = df.with_columns(h_nunique=pl.Series([3, 1, 3], dtype=pl.UInt32))
    assert_frame_equal(res_nunique, expected_nunique)

    df_bool = DataFrame({"x": [True, True], "y": [True, False]})
    p_all = Pipeline().horizontal.all(["x", "y"], value_name="h_all")
    res_all = p_all.transform(df_bool)
    expected_all = df_bool.with_columns(h_all=pl.Series([True, False]))
    assert_frame_equal(res_all, expected_all)
