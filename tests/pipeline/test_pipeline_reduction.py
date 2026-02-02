import polars as pl
import pytest
from polars import DataFrame

from polars_ml.pipeline.pipeline import Pipeline


def test_pca():
    df = DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.5, 2.5, 3.5, 4.5, 5.5],
            "x3": [0.1, 0.2, 0.1, 0.2, 0.1],
        }
    )

    pipeline = Pipeline().reduction.pca(["pc1", "pc2"], n_components=2)

    result = pipeline.fit_transform(df)

    assert result.columns == ["pc1", "pc2"]
    assert len(result) == 5
    assert result.dtypes == [pl.Float64, pl.Float64]


def test_pca_selection():
    df = DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "x2": [1.5, 2.5, 3.5],
            "y": [0, 1, 0],
        }
    )

    # Only use x1 and x2 for PCA
    pipeline = Pipeline().reduction.pca("pc1", features=["x1", "x2"], n_components=1)

    result = pipeline.fit_transform(df)

    assert result.columns == ["pc1"]


def test_umap():
    umap_installed = True
    try:
        import umap
    except ImportError:
        umap_installed = False

    if not umap_installed:
        pytest.skip("umap-learn is not installed")

    df = DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.5, 2.5, 3.5, 4.5, 5.5],
            "x3": [0.1, 0.2, 0.1, 0.2, 0.1],
        }
    )

    pipeline = Pipeline().reduction.umap(["u1", "u2"], n_components=2, n_neighbors=2)

    result = pipeline.fit_transform(df)

    assert result.columns == ["u1", "u2"]
    assert len(result) == 5
