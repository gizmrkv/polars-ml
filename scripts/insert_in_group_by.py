from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal

from const import (
    END_INSERTION_MARKER,
    GROUP_BY_NAMESPACES,
    METRICS_TRANSFORMERS,
    START_INSERTION_MARKER,
)
from polars import DataFrame
from polars.dataframe.group_by import DynamicGroupBy, GroupBy, RollingGroupBy
from utils import insert_between_markers, render_call_args, render_params_sig


def render_methods(
    group_by_cls: type[GroupBy] | type[DynamicGroupBy] | type[RollingGroupBy],
) -> list[str]:
    codes = []
    template = """
    def {name}({params}) -> Pipeline:
        return self.pipeline.pipe(GroupByGetAttr(self.attr, "{name}", self.args, self.kwargs, {call_args}))
"""
    for name, obj in inspect.getmembers(group_by_cls):
        if (
            name.startswith("_")
            or not callable(obj)
            or inspect.signature(obj).return_annotation not in {"DataFrame"}
        ):
            continue

        params = ", ".join(["self"] + render_params_sig(obj))
        call_args = ", ".join(render_call_args(obj))
        codes.append(template.format(name=name, params=params, call_args=call_args))

    return codes


def insert_in_group_by():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/group_by.py")
    for (_, namespace), cls in zip(
        GROUP_BY_NAMESPACES, [GroupBy, DynamicGroupBy, RollingGroupBy]
    ):
        codes = render_methods(cls)
        insert_between_markers(
            target_file,
            "".join(codes),
            START_INSERTION_MARKER.format(
                prefix=" " * 4, suffix=f" IN {namespace.__name__}"
            ),
            END_INSERTION_MARKER.format(
                prefix=" " * 4, suffix=f" IN {namespace.__name__}"
            ),
        )


if __name__ == "__main__":
    insert_in_group_by()
