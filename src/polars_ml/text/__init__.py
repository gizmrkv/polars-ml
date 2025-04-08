from pathlib import Path
from typing import TYPE_CHECKING

from .tfidf_truncated_svd import TfidfTruncatedSVD

if TYPE_CHECKING:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    from polars_ml import Pipeline


class TextNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def tfidf_truncated_svd(
        self,
        text_column: str,
        tfidf: "TfidfVectorizer",
        svd: "TruncatedSVD",
        *,
        prefix: str = "tfidf_truncated_svd",
        include_input: bool = True,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            TfidfTruncatedSVD(
                text_column,
                tfidf,
                svd,
                prefix=prefix,
                include_input=include_input,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )
