from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from polars.dataframe.group_by import DynamicGroupBy, GroupBy, RollingGroupBy
from utils import (
    MethodMeta,
    generate_wrapper_methods,
    insert_between_markers,
    render_call_args,
    render_params_sig,
)

from polars_ml.pipeline.group_by import (
    DynamicGroupByNameSpace,
    GroupByNameSpace,
    RollingGroupByNameSpace,
)


def is_groupby_method(name: str, obj: Any, target: Any) -> bool:
    if not callable(obj) or name.startswith("_"):
        return False
    return inspect.signature(obj).return_annotation == "DataFrame"


def extract_groupby_meta(name: str, obj: Any, target: Any) -> MethodMeta:
    params = render_params_sig(obj)
    call_args = render_call_args(obj)
    return MethodMeta(
        name=name, params_str=", ".join(params), call_args_str=", ".join(call_args)
    )


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    START_INSERTION_MARKER = "# --- START INSERTION MARKER"
    END_INSERTION_MARKER = "# --- END INSERTION MARKER"

    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/group_by.py")
    GROUPBY_TEMPLATE = """
    def {name}({params}) -> Pipeline:
        return self.pipeline.pipe(GroupByGetAttr(self.attr, "{name}", self.args, self.kwargs, {call_args}))
    """

    for ns, cls in (
        (GroupByNameSpace, GroupBy),
        (DynamicGroupByNameSpace, DynamicGroupBy),
        (RollingGroupByNameSpace, RollingGroupBy),
    ):
        codes = generate_wrapper_methods(
            targets=[cls],
            predicate=is_groupby_method,
            extractor=extract_groupby_meta,
            template=GROUPBY_TEMPLATE,
        )
        insert_between_markers(
            target_file,
            "".join(codes),
            "    " + START_INSERTION_MARKER + " IN " + ns.__name__,
            "    " + END_INSERTION_MARKER + " IN " + ns.__name__,
        )
