import unittest

import numpy as np
import polars.selectors as cs
from polars import DataFrame

from polars_ml import Pipeline
from polars_ml.testing import assert_component


class TestPipelinePlot(unittest.TestCase):
    def setUp(self):
        size = 10
        self.input_data = DataFrame(
            {
                "f0": np.random.randn(size),
                "f1": np.random.randn(size),
                "c0": np.random.choice(["a", "b"], size),
                "c1": np.random.choice(["a", "b"], size),
            }
        )

    def test_plot_corr(self):
        assert_component(
            Pipeline().select(cs.float()).plot.corr(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.select(cs.float()),
        )

    def test_plot_null(self):
        assert_component(
            Pipeline().plot.null(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_strip(self):
        assert_component(
            Pipeline().plot.strip(x=cs.string(), y=cs.float(), hue=cs.string()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_swarm(self):
        assert_component(
            Pipeline().plot.swarm(x=cs.string(), y=cs.float(), hue=cs.string()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_box(self):
        assert_component(
            Pipeline().plot.box(x=cs.string(), y=cs.float(), hue=cs.string()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_violin(self):
        assert_component(
            Pipeline().plot.violin(x=cs.string(), y=cs.float(), hue=cs.string()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_boxen(self):
        assert_component(
            Pipeline().plot.boxen(x=cs.string(), y=cs.float(), hue=cs.string()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_point(self):
        assert_component(
            Pipeline().plot.point(x=cs.string(), y=cs.float(), hue=cs.string()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_bar(self):
        assert_component(
            Pipeline().plot.bar(x=cs.string(), y=cs.float(), hue=cs.string()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_count(self):
        assert_component(
            Pipeline().plot.count(x=cs.string(), hue=cs.string()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_hist(self):
        assert_component(
            Pipeline().plot.hist(x=cs.float(), hue=cs.string()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_kde(self):
        assert_component(
            Pipeline().plot.kde(x=cs.float(), hue=cs.string()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_ecdf(self):
        assert_component(
            Pipeline().plot.ecdf(x=cs.float(), hue=cs.string()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_scatter(self):
        assert_component(
            Pipeline().plot.scatter(x=cs.float(), y=cs.float(), hue=cs.all()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_plot_line(self):
        assert_component(
            Pipeline().plot.line(x=cs.float(), y=cs.float(), hue=cs.string()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )
