from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Self

import numpy as np
from polars import DataFrame, Series

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfTruncatedSVD(PipelineComponent):
    def __init__(
        self,
        text_column: str,
        tfidf: "TfidfVectorizer",
        svd: "TruncatedSVD",
        *,
        prefix: str = "tfidf_truncated_svd",
        include_input: bool = True,
        out_dir: str | Path | None = None,
    ):
        self.text_column = text_column
        self.tfidf = tfidf
        self.svd = svd
        self.prefix = prefix
        self.include_input = include_input
        self.out_dir = Path(out_dir) if out_dir is not None else None

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        text_data = data[self.text_column].to_list()
        tfidf_matrix = self.tfidf.fit_transform(text_data)
        self.feature_names = self.tfidf.get_feature_names_out()
        self.svd.fit(tfidf_matrix)

        if self.out_dir is not None:
            self.save()

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        text_data = data[self.text_column].to_list()
        tfidf_matrix = self.tfidf.transform(text_data)
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

        joblib.dump(self.tfidf, out_dir / "tfidf_vectorizer.pkl")
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
