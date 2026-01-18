from __future__ import annotations

import inspect
from inspect import Parameter
from pathlib import Path
from typing import Any, Callable, Mapping


def render_param_sig(
    param: Parameter, *, override_annotation: str | None = None
) -> str:
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
    *,
    override_annotations: Mapping[str, str] | None = None,
    skip_params: set[str] | None = None,
) -> list[str]:
    params = list(inspect.signature(func).parameters.values())
    anns = override_annotations or {}
    skip = skip_params or set()
    return [
        render_param_sig(p, override_annotation=anns.get(p.annotation))
        for p in params
        if p.name not in {"self", "cls"} | skip
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
) -> None:
    """
    Inserts (replaces) text between a start marker and an end marker in the specified file.

    Args:
        file_path (str | Path): Path to the target text file.
        text_to_insert (str): The text content to insert.
        start_marker (str): The text identifying the start line.
        end_marker (str): The text identifying the end line.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If markers are missing or the order is invalid.
        IOError: If reading or writing fails.
    """

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
