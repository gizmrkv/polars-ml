from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence

from polars._typing import ColumnNameOrSelector

from .sk_learn_liner import SKLearnLinearModel

if TYPE_CHECKING:
    from polars_ml.pipeline.pipeline import Pipeline


class LinearNameSpace:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def linear_regression(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import LinearRegression

        return self.pipeline.pipe(
            SKLearnLinearModel(
                LinearRegression(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def ridge(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import Ridge

        return self.pipeline.pipe(
            SKLearnLinearModel(
                Ridge(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def ridge_cv(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import RidgeCV

        return self.pipeline.pipe(
            SKLearnLinearModel(
                RidgeCV(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def lasso(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import Lasso

        return self.pipeline.pipe(
            SKLearnLinearModel(
                Lasso(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def lasso_cv(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import LassoCV

        return self.pipeline.pipe(
            SKLearnLinearModel(
                LassoCV(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def elastic_net(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import ElasticNet

        return self.pipeline.pipe(
            SKLearnLinearModel(
                ElasticNet(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def elastic_net_cv(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import ElasticNetCV

        return self.pipeline.pipe(
            SKLearnLinearModel(
                ElasticNetCV(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def logistic_regression(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import LogisticRegression

        return self.pipeline.pipe(
            SKLearnLinearModel(
                LogisticRegression(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def logistic_regression_cv(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import LogisticRegressionCV

        return self.pipeline.pipe(
            SKLearnLinearModel(
                LogisticRegressionCV(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def sgd_regressor(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import SGDRegressor

        return self.pipeline.pipe(
            SKLearnLinearModel(
                SGDRegressor(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def sgd_classifier(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import SGDClassifier

        return self.pipeline.pipe(
            SKLearnLinearModel(
                SGDClassifier(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def huber_regressor(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import HuberRegressor

        return self.pipeline.pipe(
            SKLearnLinearModel(
                HuberRegressor(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def quantile_regressor(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import QuantileRegressor

        return self.pipeline.pipe(
            SKLearnLinearModel(
                QuantileRegressor(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def passive_aggressive_regressor(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import PassiveAggressiveRegressor

        return self.pipeline.pipe(
            SKLearnLinearModel(
                PassiveAggressiveRegressor(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def passive_aggressive_classifier(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import PassiveAggressiveClassifier

        return self.pipeline.pipe(
            SKLearnLinearModel(
                PassiveAggressiveClassifier(**kwargs),
                target,
                prediction,
                features,
            )
        )

    def perceptron(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        from sklearn.linear_model import Perceptron

        return self.pipeline.pipe(
            SKLearnLinearModel(
                Perceptron(**kwargs),
                target,
                prediction,
                features,
            )
        )
