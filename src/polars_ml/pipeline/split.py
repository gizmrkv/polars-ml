from abc import ABC
from typing import TYPE_CHECKING, Generic, Iterable

from polars._typing import IntoExpr

from polars_ml.split import KFold, TrainValidSplit
from polars_ml.typing import PipelineType

if TYPE_CHECKING:
    from .lazy_pipeline import LazyPipeline  # noqa: F401
    from .pipeline import Pipeline  # noqa: F401


class BaseSplitNameSpace(Generic[PipelineType], ABC):
    def __init__(self, pipeline: PipelineType):
        self.pipeline = pipeline

    def train_valid_split(
        self,
        test_size: float,
        *,
        split_name: str = "is_valid",
        stratify: str | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> PipelineType:
        return self.pipeline.pipe(
            TrainValidSplit(
                test_size=test_size,
                split_name=split_name,
                stratify=stratify,
                shuffle=shuffle,
                seed=seed,
            )
        )

    def k_fold(
        self,
        n_splits: int = 5,
        *,
        split_name: str = "fold",
        shuffle: bool = True,
        seed: int | None = None,
    ) -> PipelineType:
        return self.pipeline.pipe(
            KFold(
                n_splits=n_splits,
                split_name=split_name,
                shuffle=shuffle,
                seed=seed,
            )
        )

    def stratified_k_fold(
        self,
        n_splits: int = 5,
        *,
        split_name: str = "fold",
        stratify: IntoExpr | Iterable[IntoExpr],
        shuffle: bool = True,
        seed: int | None = None,
    ) -> PipelineType:
        return self.pipeline.pipe(
            KFold(
                n_splits=n_splits,
                split_name=split_name,
                stratify=stratify,
                shuffle=shuffle,
                seed=seed,
            )
        )


class SplitNameSpace(BaseSplitNameSpace["Pipeline"]):
    pass


class LazySplitNameSpace(BaseSplitNameSpace["LazyPipeline"]):
    pass
