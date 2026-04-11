from __future__ import annotations

import inspect
from dataclasses import dataclass
from inspect import Parameter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import polars as pl
from polars.dataframe.group_by import DynamicGroupBy, GroupBy, RollingGroupBy

from polars_ml.pipeline.group_by import (
    DynamicGroupByNameSpace,
    GroupByNameSpace,
    RollingGroupByNameSpace,
)


@dataclass
class MethodMeta:
    name: str
    params_str: str
    call_args_str: str


def render_param_sig(param: Parameter, override_annotation: str | None = None) -> str:
    if param.kind is Parameter.VAR_POSITIONAL:
        s = f"*{param.name}"
    elif param.kind is Parameter.VAR_KEYWORD:
        s = f"**{param.name}"
    else:
        s = param.name

    if override_annotation is not None:
        s += f": {override_annotation}"
    elif param.annotation is not Parameter.empty:
        s += f": {str(param.annotation)}"

    if (
        param.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
        and param.default is not Parameter.empty
    ):
        s += f" = {repr(param.default)}"

    return s


def render_params_sig(
    func: Callable[..., Any],
    override_annotations: Mapping[str, str] | None = None,
) -> list[str]:
    params = list(inspect.signature(func).parameters.values())
    anns = override_annotations or {}
    return [
        render_param_sig(p, override_annotation=anns.get(p.annotation)) for p in params
    ]


def render_call_arg(param: Parameter) -> str:
    if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
        return param.name
    elif param.kind is Parameter.VAR_POSITIONAL:
        return f"*{param.name}"
    elif param.kind is Parameter.KEYWORD_ONLY:
        return f"{param.name}={param.name}"
    elif param.kind is Parameter.VAR_KEYWORD:
        return f"**{param.name}"
    else:
        raise ValueError(f"Unknown parameter kind: {param.kind}")


def render_call_args(
    func: Callable[..., Any], skip_params: set[str] | None = None
) -> list[str]:
    params = list(inspect.signature(func).parameters.values())
    skip = skip_params or set()
    return [render_call_arg(p) for p in params if p.name not in {"self", "cls"} | skip]


def insert_between_markers(
    filepath: str | Path, text: str, start_marker: str, end_marker: str
):
    target_path = Path(filepath)

    if not target_path.exists():
        raise FileNotFoundError(f"File not found: {target_path}")

    with target_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    start_index = -1
    end_index = -1

    for i, line in enumerate(lines):
        if start_marker in line and start_index == -1:
            start_index = i
        elif end_marker in line and end_index == -1:
            end_index = i

        if start_index != -1 and end_index != -1:
            break

    missing: list[str] = []
    if start_index == -1:
        missing.append(f"Start marker '{start_marker}'")
    if end_index == -1:
        missing.append(f"End marker '{end_marker}'")

    if missing:
        raise ValueError(f"Markers not found: {', '.join(missing)}")

    if start_index >= end_index:
        raise ValueError(
            f"Invalid marker order (Start line: {start_index + 1}, End line: {end_index + 1})"
        )

    if not text.endswith("\n"):
        text += "\n"

    new_lines = lines[: start_index + 1] + [text] + lines[end_index:]

    with target_path.open("w", encoding="utf-8") as f:
        f.writelines(new_lines)


def generate_wrapper_methods(
    targets: Iterable[Any],
    predicate: Callable[[str, Any, Any], bool],
    extractor: Callable[[str, Any, Any], MethodMeta],
    template: str,
) -> list[str]:
    codes = []
    for target in targets:
        for name, obj in inspect.getmembers(target):
            if not predicate(name, obj, target):
                continue

            meta = extractor(name, obj, target)
            codes.append(
                template.format(
                    name=meta.name, params=meta.params_str, call_args=meta.call_args_str
                )
            )
    return codes


# ==========================================
# Pipeline
# ==========================================
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


# ==========================================
# LazyPipeline
# ==========================================
def is_lazy_pipeline_method(name: str, obj: Any, target: Any) -> bool:
    if (
        not callable(obj)
        or name.startswith("_")
        or name in {"scan_iceberg", "scan_pyarrow_dataset"}
        or (target is pl and not name.startswith("scan_"))
    ):
        return False

    sig = inspect.signature(obj)
    if sig.return_annotation not in ("LazyFrame", "Self") and (
        not name.startswith("scan_")
    ):
        return False

    return True


def extract_lazy_pipeline_meta(name: str, obj: Any, target: Any) -> MethodMeta:
    params = [
        render_param_sig(
            p,
            "pl.LazyFrame | LazyTransformer" if p.annotation == "LazyFrame" else None,
        )
        for p in inspect.signature(obj).parameters.values()
    ]
    if not params or not params[0].startswith("self"):
        params.insert(0, "self")

    call_args = [
        f'"{name}"',
        "None" if target is pl.LazyFrame else "pl",
        *render_call_args(obj),
    ]

    return MethodMeta(
        name=name, params_str=", ".join(params), call_args_str=", ".join(call_args)
    )


# ==========================================
# GroupBy
# ==========================================
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

    # --- Pipeline ---
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

    # --- LazyPipeline ---
    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/pipeline_lazy.py")
    LAZY_PIPELINE_TEMPLATE = """
    def {name}({params}) -> Self:
        return self.pipe(LazyGetAttr({call_args}))
"""
    codes = generate_wrapper_methods(
        targets=[pl.LazyFrame, pl],
        predicate=is_lazy_pipeline_method,
        extractor=extract_lazy_pipeline_meta,
        template=LAZY_PIPELINE_TEMPLATE,
    )
    insert_between_markers(
        target_file,
        "".join(codes),
        "    " + START_INSERTION_MARKER + " IN LazyPipeline",
        "    " + END_INSERTION_MARKER + " IN LazyPipeline",
    )

    # --- GroupBy ---
    # target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/group_by.py")
    # GROUPBY_TEMPLATE = """
    # def {name}({params}) -> GroupByGetAttr:
    # return self.pipeline.pipe(GroupByGetAttr(self.attr, "{name}", self.args, self.kwargs, {call_args}))
    # """

    # for ns, cls in (
    #     (GroupByNameSpace, GroupBy),
    #     (DynamicGroupByNameSpace, DynamicGroupBy),
    #     (RollingGroupByNameSpace, RollingGroupBy),
    # ):
    #     codes = generate_wrapper_methods(
    #         targets=[cls],
    #         predicate=is_groupby_method,
    #         extractor=extract_groupby_meta,
    #         template=GROUPBY_TEMPLATE
    #     )
    #     insert_between_markers(
    #         target_file,
    #         "".join(codes),
    #         "    " + START_INSERTION_MARKER + " IN " + ns.__name__,
    #         "    " + END_INSERTION_MARKER + " IN " + ns.__name__,
    #     )
