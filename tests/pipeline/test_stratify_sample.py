import polars as pl
from polars.testing import assert_frame_equal

from polars_ml.pipeline.stratify_sample import StratifySample


def test_stratify_sample_basic() -> None:
    df = pl.DataFrame({"g": ["a"] * 100 + ["b"] * 100, "v": range(200)})

    # Sample 10% from each group
    sampler = StratifySample(by="g", fraction=0.1)
    # StratifySample is a Transformer, not LazyTransformer
    sampled = sampler.transform(df)

    # 10% of 100 is 10. Total 20 rows.
    assert sampled.height == 20
    assert (sampled["g"] == "a").sum() == 10
    assert (sampled["g"] == "b").sum() == 10


def test_stratify_sample_seed() -> None:
    df = pl.DataFrame({"g": ["a"] * 10 + ["b"] * 10, "v": range(20)})

    sampler = StratifySample(by="g", fraction=0.5, seed=42, maintain_order=True)

    sampled1 = sampler.transform(df)
    sampled2 = sampler.transform(df)

    assert_frame_equal(sampled1, sampled2)


def test_stratify_sample_multiple_columns() -> None:
    df = pl.DataFrame(
        {"g1": ["x", "x", "y", "y"], "g2": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]}
    )

    # 1 row per group (since each group is size 1 and fraction is 1.0 or high enough)
    sampler = StratifySample(by=["g1", "g2"], fraction=1.0)
    sampled = sampler.transform(df)

    assert sampled.height == 4
    # Check that all combinations are present
    assert sampled.sort("g1", "g2")["v"].to_list() == [1, 2, 3, 4]
