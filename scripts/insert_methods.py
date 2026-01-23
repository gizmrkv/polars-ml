from __future__ import annotations

import inspect
from inspect import Parameter
from pathlib import Path
from typing import Any, Callable, Mapping

import polars as pl
from polars import DataFrame
from polars.dataframe.group_by import DynamicGroupBy, GroupBy, RollingGroupBy

from polars_ml.pipeline.group_by import (
    DynamicGroupByNameSpace,
    GroupByNameSpace,
    RollingGroupByNameSpace,
)


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
    param = param
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

    # Search for marker positions
    for i, line in enumerate(lines):
        if start_marker in line and start_index == -1:
            start_index = i
        elif end_marker in line and end_index == -1:
            end_index = i

        if start_index != -1 and end_index != -1:
            break

    # Error check
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


START_INSERTION_MARKER = "# --- START INSERTION MARKER"
END_INSERTION_MARKER = "# --- END INSERTION MARKER"


def render_pipeline_methods() -> list[str]:
    codes: list[str] = []

    template = """
    def {name}({params}) -> Self:
        return self.pipe(GetAttr({call_args}))
"""
    for object in [DataFrame, pl]:
        for name, obj in inspect.getmembers(object):
            if (
                not callable(obj)
                or name.startswith("_")
                or name in {"map_columns", "write_iceberg"}
                or (object is pl and not name.startswith("read_"))
            ):
                continue

            sig = inspect.signature(obj)
            if sig.return_annotation not in ("DataFrame", "Self") and (
                not name.startswith("write_")
            ):
                continue

            params = [
                render_param_sig(
                    p,
                    "DataFrame | Transformer" if p.annotation == "DataFrame" else None,
                )
                for p in inspect.signature(obj).parameters.values()
            ]
            if params[0] != "self":
                params = ["self"] + params

            codes.append(
                template.format(
                    name=name,
                    params=",".join(params),
                    call_args=",".join(
                        [
                            f'"{name}"',
                            "None" if object is DataFrame else "pl",
                            *render_call_args(obj),
                        ]
                    ),
                )
            )

    return codes


def render_group_by_methods(
    group_by_cls: type[GroupBy] | type[DynamicGroupBy] | type[RollingGroupBy],
) -> list[str]:
    codes = []
    template = """
    def {name}({params}) -> Pipeline:
        return self.pipeline.pipe(GroupByGetAttr(self.attr, "{name}", self.args, self.kwargs, {call_args}))
"""
    for name, obj in inspect.getmembers(group_by_cls):
        if not (
            callable(obj)
            and not name.startswith("_")
            and inspect.signature(obj).return_annotation == "DataFrame"
        ):
            continue

        codes.append(
            template.format(
                name=name,
                params=",".join(render_params_sig(obj)),
                call_args=",".join(render_call_args(obj)),
            )
        )

    return codes


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/pipeline.py")
    codes = render_pipeline_methods()
    insert_between_markers(
        target_file,
        "".join(codes),
        "    " + START_INSERTION_MARKER + " IN Pipeline",
        "    " + END_INSERTION_MARKER + " IN Pipeline",
    )

    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/group_by.py")
    for ns, cls in (
        (GroupByNameSpace, GroupBy),
        (DynamicGroupByNameSpace, DynamicGroupBy),
        (RollingGroupByNameSpace, RollingGroupBy),
    ):
        codes = render_group_by_methods(cls)
        insert_between_markers(
            target_file,
            "".join(codes),
            "    " + START_INSERTION_MARKER + " IN " + ns.__name__,
            "    " + END_INSERTION_MARKER + " IN " + ns.__name__,
        )
