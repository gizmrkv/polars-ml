from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from polars.lazyframe.group_by import LazyGroupBy
from utils import (
    MethodMeta,
    generate_wrapper_methods,
    insert_between_markers,
    render_call_args,
    render_params_sig,
)


def is_lazy_groupby_method(name: str, obj: Any, target: Any) -> bool:
    if not callable(obj) or name.startswith("_"):
        return False
    return inspect.signature(obj).return_annotation == "LazyFrame"


def extract_lazy_groupby_meta(name: str, obj: Any, target: Any) -> MethodMeta:
    params = render_params_sig(obj)
    call_args = render_call_args(obj)
    return MethodMeta(
        name=name, params_str=", ".join(params), call_args_str=", ".join(call_args)
    )


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    START_INSERTION_MARKER = "# --- START INSERTION MARKER"
    END_INSERTION_MARKER = "# --- END INSERTION MARKER"

    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/group_by_lazy.py")
    LAZY_GROUPBY_TEMPLATE = """
    def {name}({params}) -> LazyPipeline:
        return self.pipeline.pipe(LazyGroupByGetAttr(self.attr, "{name}", self.args, self.kwargs, {call_args}))
    """
    codes = generate_wrapper_methods(
        targets=[LazyGroupBy],
        predicate=is_lazy_groupby_method,
        extractor=extract_lazy_groupby_meta,
        template=LAZY_GROUPBY_TEMPLATE,
    )
    insert_between_markers(
        target_file,
        "".join(codes),
        "    " + START_INSERTION_MARKER + " IN LazyGroupByNameSpace",
        "    " + END_INSERTION_MARKER + " IN LazyGroupByNameSpace",
    )
