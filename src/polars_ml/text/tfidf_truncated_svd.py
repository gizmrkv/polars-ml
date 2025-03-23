from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Self, TypedDict

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame, Series
from polars._typing import IntoExpr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from polars_ml.pipeline.component import PipelineComponent


class TfidfParameters(TypedDict, total=False):
    encoding: str
    decode_error: Literal["strict", "ignore", "replace"]
    strip_accents: Literal["ascii", "unicode"] | Callable[..., Any] | None
    lowercase: bool
    preprocessor: Callable[..., Any] | None
    tokenizer: Callable[..., Any] | None
    analyzer: Literal["word", "char", "char_wb"] | Callable[..., Any]
    stop_words: Literal["english"] | list[str] | None
    token_pattern: str
    ngram_range: tuple[int, int]
    max_df: float | int
    min_df: float | int
    max_features: int | None
    vocabulary: Mapping[str, int] | Iterable[str] | None
    binary: bool
    norm: Literal["l1", "l2"] | None
    use_idf: bool
    smooth_idf: bool
    sublinear_tf: bool


class TruncatedSVDParameters(TypedDict, total=False):
    n_components: int
    algorithm: Literal["arpack", "randomized"]
    n_iter: int
    n_oversamples: int
    power_iteration_normalizer: Literal["auto", "QR", "LU", "none"]
    random_state: int | None
    tol: float


class TfidfTruncatedSVD(PipelineComponent):
    def __init__(
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
    ):
        self.text_column = text_column
        self.prefix = prefix
        self.include_input = include_input
        self.tfidf_kwargs = tfidf_kwargs or {}
        self.svd_kwargs = svd_kwargs or {}
        self.out_dir = Path(out_dir) if out_dir is not None else None

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        text_data = data[self.text_column].to_list()

        tfidf_kwargs = (
            self.tfidf_kwargs(data)
            if callable(self.tfidf_kwargs)
            else self.tfidf_kwargs
        )

        svd_kwargs = (
            self.svd_kwargs(data) if callable(self.svd_kwargs) else self.svd_kwargs
        )
        self.vectorizer = TfidfVectorizer(**tfidf_kwargs)
        tfidf_matrix = self.vectorizer.fit_transform(text_data)

        self.feature_names = self.vectorizer.get_feature_names_out()

        self.svd = TruncatedSVD(**svd_kwargs)
        self.svd.fit(tfidf_matrix)

        if self.out_dir is not None:
            self.save()

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        text_data = data[self.text_column].to_list()

        tfidf_matrix = self.vectorizer.transform(text_data)

        transformed = self.svd.transform(tfidf_matrix)

        svd_columns = [
            Series(f"{self.prefix}_{i}", transformed[:, i])
            for i in range(transformed.shape[1])
        ]

        if self.include_input:
            return data.with_columns(svd_columns)
        else:
            return DataFrame(svd_columns)

    def save(self, out_dir: str | Path | None = None):
        out_dir = Path(out_dir) if out_dir else self.out_dir
        if out_dir is None:
            raise ValueError("No output directory provided")

        out_dir.mkdir(parents=True, exist_ok=True)

        import joblib

        joblib.dump(self.vectorizer, out_dir / "tfidf_vectorizer.pkl")
        joblib.dump(self.svd, out_dir / "truncated_svd.pkl")

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.bar(
            range(1, len(self.svd.explained_variance_ratio_) + 1),
            self.svd.explained_variance_ratio_,
        )
        plt.plot(
            range(1, len(self.svd.explained_variance_ratio_) + 1),
            np.cumsum(self.svd.explained_variance_ratio_),
            "ro-",
        )
        plt.xlabel("Components")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Explained Variance by Components")
        plt.grid(True)
        plt.savefig(out_dir / "explained_variance.png")
        plt.close()
