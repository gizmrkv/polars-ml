from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import polars as pl
from utils import (
    MethodMeta,
    generate_wrapper_methods,
    insert_between_markers,
    render_call_args,
    render_param_sig,
)


def is_pipeline_method(name: str, obj: Any, target: Any) -> bool:
    if (
        not callable(obj)
        or name.startswith("_")
        or name in {"map_columns", "write_iceberg"}
        or (target is pl and not name.startswith("read_"))
    ):
        return False

    sig = inspect.signature(obj)
    if sig.return_annotation not in ("DataFrame", "Self") and (
        not name.startswith("write_")
    ):
        return False

    return True


def extract_pipeline_meta(name: str, obj: Any, target: Any) -> MethodMeta:
    params = [
        render_param_sig(
            p,
            "pl.DataFrame | Transformer" if p.annotation == "DataFrame" else None,
        )
        for p in inspect.signature(obj).parameters.values()
    ]
    if not params or not params[0].startswith("self"):
        params.insert(0, "self")

    call_args = [
        f'"{name}"',
        "None" if target is pl.DataFrame else "pl",
        *render_call_args(obj),
    ]

    return MethodMeta(
        name=name, params_str=", ".join(params), call_args_str=", ".join(call_args)
    )


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    START_INSERTION_MARKER = "# --- START INSERTION MARKER"
    END_INSERTION_MARKER = "# --- END INSERTION MARKER"

    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/pipeline.py")
    PIPELINE_TEMPLATE = """
    def {name}({params}) -> Self:
        return self.pipe(GetAttr({call_args}))
"""
    codes = generate_wrapper_methods(
        targets=[pl.DataFrame, pl],
        predicate=is_pipeline_method,
        extractor=extract_pipeline_meta,
        template=PIPELINE_TEMPLATE,
    )
    insert_between_markers(
        target_file,
        "".join(codes),
        "    " + START_INSERTION_MARKER + " IN Pipeline",
        "    " + END_INSERTION_MARKER + " IN Pipeline",
    )
