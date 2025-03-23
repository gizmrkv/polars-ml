from pathlib import Path
from typing import TYPE_CHECKING, Callable

from polars import DataFrame

from .tfidf_truncated_svd import (
    TfidfParameters,
    TfidfTruncatedSVD,
    TruncatedSVDParameters,
)

if TYPE_CHECKING:
    from polars_ml import Pipeline


class TextNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def tfidf_truncated_svd(
        self,
        text_column: str,
        *,
        prefix: str = "tfidf_truncated_svd",
        include_input: bool = True,
        tfidf_kwargs: TfidfParameters
        | Callable[[DataFrame], TfidfParameters]
        | None = None,
        svd_kwargs: TruncatedSVDParameters
        | Callable[[DataFrame], TruncatedSVDParameters]
        | None = None,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            TfidfTruncatedSVD(
                text_column,
                prefix=prefix,
                include_input=include_input,
                tfidf_kwargs=tfidf_kwargs,
                svd_kwargs=svd_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )
