import random
from typing import Iterator, Mapping, TypeVar

T = TypeVar("T")


def incremental_sampling(n_rows: int, n_blocks: int) -> Iterator[list[int]]:
    if n_rows < 2**n_blocks:
        raise ValueError("n_rows must be less than or equal to 2 ** n_subsamples")

    indexes = list(range(n_rows))
    random.shuffle(indexes)
    for i in range(n_blocks - 1, -1, -1):
        yield indexes[: n_rows >> i]


def deduplicate_scores(
    scores: Mapping[T, float], tolerance: float = 1e-14
) -> dict[T, float]:
    name_scores = sorted(scores.items(), key=lambda x: x[1])
    deduplicated_scores = {name_scores[0][0]: name_scores[0][1]}
    prev_score = name_scores[0][1] + 2 * tolerance
    for name, score in name_scores:
        if abs(score - prev_score) > tolerance:
            deduplicated_scores[name] = score
            prev_score = score

    return deduplicated_scores
