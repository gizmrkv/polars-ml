import unittest

import polars as pl
import polars.selectors as cs
from polars import DataFrame, Series

from polars_ml import Pipeline
from polars_ml.testing import assert_component


class TestPipelineHorizontal(unittest.TestCase):
    def setUp(self):
        self.input_data = DataFrame(
            [
                Series("i0", [0, 1, None, 0, 0, None, 0, None]),
                Series("i1", [-1, 0, 10, None, 0, None, 0, 0]),
                Series("i2", [0, -10, 0, 100, None, 0, 0, 0]),
                Series("i3", [0, 0, -100, 0, 1000, None, 0, 0]),
                Series("i4", [0, 0, 0, -1000, 0, 10000, None, 0]),
                Series("b0", [0, 1, 1, 0, 0, 1, 0, 0]).cast(pl.Boolean),
                Series("b1", [0, 1, 0, 1, 0, 1, 0, 0]).cast(pl.Boolean),
                Series("b2", [0, 1, 0, 0, 1, 1, 0, 0]).cast(pl.Boolean),
            ]
        )

    def test_horizontal_agg(self):
        assert_component(
            Pipeline()
            .horizontal.agg(cs.starts_with("i"), aggs=[(pl.col("value") ** 2).mean()])
            .select("value"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series(
                        "value",
                        [0.2, 20.2, 2525.0, 252500.0, 250000.0, 50000000.0, 0.0, 0.0],
                        dtype=pl.Float64,
                    )
                ]
            ),
        )

    def test_horizontal_all(self):
        assert_component(
            Pipeline().horizontal.all(cs.starts_with("b")).select("all"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [Series("all", [False, True, False, False, False, True, False, False])]
            ),
        )

    def test_horizontal_any(self):
        assert_component(
            Pipeline().horizontal.any(cs.starts_with("b")).select("any"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [Series("any", [False, True, True, True, True, True, False, False])]
            ),
        )

    def test_horizontal_count(self):
        assert_component(
            Pipeline().horizontal.count(cs.starts_with("i")).select("count"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [Series("count", [5, 5, 4, 4, 4, 2, 4, 4], dtype=pl.UInt32)]
            ),
        )

    def test_horizontal_null_count(self):
        assert_component(
            Pipeline().horizontal.null_count(cs.starts_with("i")).select("null_count"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [Series("null_count", [0, 0, 1, 1, 1, 3, 1, 1], dtype=pl.UInt32)]
            ),
        )

    def test_horizontal_n_unique(self):
        assert_component(
            Pipeline().horizontal.n_unique(cs.starts_with("i")).select("n_unique"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [Series("n_unique", [2, 3, 4, 4, 3, 3, 2, 2], dtype=pl.UInt32)]
            ),
        )

    def test_horizontal_max(self):
        assert_component(
            Pipeline().horizontal.max(cs.starts_with("i")).select("max"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [Series("max", [0, 1, 10, 100, 1000, 10000, 0, 0], dtype=pl.Int64)]
            ),
        )

    def test_horizontal_min(self):
        assert_component(
            Pipeline().horizontal.min(cs.starts_with("i")).select("min"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [Series("min", [-1, -10, -100, -1000, 0, 0, 0, 0], dtype=pl.Int64)]
            ),
        )

    def test_horizontal_nan_max(self):
        assert_component(
            Pipeline().horizontal.nan_max(cs.starts_with("i")).select("nan_max"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series(
                        "nan_max",
                        [0, 1, 10, 100, 1000, 10000, 0, 0],
                        dtype=pl.Int64,
                    )
                ]
            ),
        )

    def test_horizontal_nan_min(self):
        assert_component(
            Pipeline().horizontal.nan_min(cs.starts_with("i")).select("nan_min"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series(
                        "nan_min",
                        [-1, -10, -100, -1000, 0, 0, 0, 0],
                        dtype=pl.Int64,
                    )
                ]
            ),
        )

    def test_horizontal_arg_max(self):
        assert_component(
            Pipeline().horizontal.arg_max(cs.starts_with("i")).select("arg_max"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series(
                        "arg_max",
                        [
                            ["i0", "i2", "i3", "i4"],
                            ["i0"],
                            ["i1"],
                            ["i2"],
                            ["i3"],
                            ["i4"],
                            ["i0", "i1", "i2", "i3"],
                            ["i1", "i2", "i3", "i4"],
                        ],
                    )
                ]
            ),
            check_deepcopy=False,
            check_pickle_dump_load=False,
            check_joblib_dump_load=False,
        )

    def test_horizontal_arg_min(self):
        assert_component(
            Pipeline().horizontal.arg_min(cs.starts_with("i")).select("arg_min"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series(
                        "arg_min",
                        [
                            ["i1"],
                            ["i2"],
                            ["i3"],
                            ["i4"],
                            ["i0", "i1", "i4"],
                            ["i2"],
                            ["i0", "i1", "i2", "i3"],
                            ["i1", "i2", "i3", "i4"],
                        ],
                    )
                ]
            ),
            check_deepcopy=False,
            check_pickle_dump_load=False,
            check_joblib_dump_load=False,
        )

    def test_horizontal_median(self):
        assert_component(
            Pipeline().horizontal.median(cs.starts_with("i")).select("median"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series(
                        "median",
                        [0.0, 0.0, 0.0, 0.0, 0.0, 5000.0, 0.0, 0.0],
                        dtype=pl.Float64,
                    )
                ]
            ),
        )

    def test_horizontal_quantile(self):
        assert_component(
            Pipeline()
            .horizontal.quantile(cs.starts_with("i"), quantile=0.5)
            .select("quantile"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series(
                        "quantile",
                        [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0, 0.0, 0.0],
                        dtype=pl.Float64,
                    )
                ]
            ),
        )

    def test_horizontal_mean(self):
        assert_component(
            Pipeline().horizontal.mean(cs.starts_with("i")).select("mean"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series(
                        "mean",
                        [-0.2, -1.8, -22.5, -225.0, 250.0, 5000.0, 0.0, 0.0],
                        dtype=pl.Float64,
                    )
                ]
            ),
        )

    def test_horizontal_sum(self):
        assert_component(
            Pipeline().horizontal.sum(cs.starts_with("i")).select("sum"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series(
                        "sum",
                        [-1, -9, -90, -900, 1000, 10000, 0, 0],
                        dtype=pl.Int64,
                    )
                ]
            ),
        )

    def test_horizontal_std(self):
        assert_component(
            Pipeline().horizontal.std(cs.starts_with("i")).select("std"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series(
                        "std",
                        [
                            0.447213595499958,
                            4.604345773288535,
                            51.881274720911264,
                            518.8127472091127,
                            500.00000000000006,
                            7071.067811865475,
                            0.0,
                            0.0,
                        ],
                        dtype=pl.Float64,
                    )
                ]
            ),
        )

    def test_horizontal_var(self):
        assert_component(
            Pipeline().horizontal.var(cs.starts_with("i")).select("var"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series(
                        "var",
                        [
                            0.20000000000000004,
                            21.2,
                            2691.6666666666665,
                            269166.6666666666,
                            250000.00000000003,
                            50000000.0,
                            0.0,
                            0.0,
                        ],
                        dtype=pl.Float64,
                    )
                ]
            ),
        )
