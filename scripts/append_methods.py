from __future__ import annotations

import inspect
import re
from inspect import Parameter
from pathlib import Path
from typing import Any, Callable, Mapping

import inflection
import polars as pl
from polars import DataFrame

from polars_ml.pipeline.basic import Apply, Const, Echo, Parrot, Side

_EMPTY = inspect._empty  # type: ignore


def format_param(param: Parameter, *, override_annotation: str | None = None) -> str:
    if param.kind is Parameter.VAR_POSITIONAL:
        s = f"*{param.name}"
    elif param.kind is Parameter.VAR_KEYWORD:
        s = f"**{param.name}"
    else:
        s = param.name

    if override_annotation is not None:
        s += f": {override_annotation}"
    elif param.annotation is not _EMPTY:
        s += f": {str(param.annotation)}"

    if (
        param.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
        and param.default is not _EMPTY
    ):
        s += f" = {repr(param.default)}"

    return s


def format_call_args(func: Callable[..., Any]) -> list[str]:
    sig = inspect.signature(func)

    parts: list[str] = []
    for p in sig.parameters.values():
        if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
            parts.append(p.name)
        elif p.kind is Parameter.VAR_POSITIONAL:
            parts.append(f"*{p.name}")
        elif p.kind is Parameter.KEYWORD_ONLY:
            parts.append(f"{p.name}={p.name}")
        elif p.kind is Parameter.VAR_KEYWORD:
            parts.append(f"**{p.name}")
        else:
            raise ValueError(f"Unknown parameter kind: {p.kind}")

    return parts


def get_dataframe_methods() -> list[str]:
    methods: list[str] = []
    for name, obj in inspect.getmembers(DataFrame):
        if (
            name.startswith("_")
            or not callable(obj)
            or name in {"map_columns", "deserialize"}
        ):
            continue

        ret = inspect.signature(obj).return_annotation
        if ret in {"DataFrame", "Self"}:
            methods.append(name)

    return sorted(set(methods))


def format_builtin_method_signatures() -> list[str]:
    sigs = []
    for name in get_dataframe_methods():
        method = getattr(DataFrame, name)
        sig = inspect.signature(method)
        sigs.append(
            f"""
    def {name}({", ".join(format_param(p).replace(": DataFrame", ": DataFrame | Transformer") for p in sig.parameters.values())}) -> Self:
        return self.pipe(GetAttr({", ".join([f'"{name}"'] + format_call_args(method)[1:])}))
            """
        )

    return sigs


def format_custom_method_signatures(*transformer_types: type) -> list[str]:
    sigs = []
    for type in transformer_types:
        method = getattr(type, "__init__")
        sig = inspect.signature(method)
        name = type.__name__
        sigs.append(
            f"""
    def {inflection.underscore(name)}({", ".join(format_param(p) for p in sig.parameters.values())}) -> Self:
        return self.pipe({name}({", ".join(format_call_args(method)[1:])}))
            """
        )

    return sigs


def append_methods(
    target_file: str | Path,
    class_name: str,
    code: str,
    start_marker: str = "    # --- BEGIN AUTO-GENERATED METHODS ---",
    end_marker: str = "    # --- END AUTO-GENERATED METHODS ---",
):
    target_file = Path(target_file)
    content = target_file.read_text(encoding="utf-8")
    generated_block = f"{start_marker}\n{code}\n{end_marker}"
    pattern = re.compile(
        f"{re.escape(start_marker)}.*?{re.escape(end_marker)}", re.DOTALL
    )
    if pattern.search(content):
        print("Updating existing generated methods...")
        new_content = pattern.sub(generated_block, content)
    else:
        print("Appending new generated methods...")
        if f"class {class_name}(" in content:
            new_content = content.rstrip() + "\n\n" + generated_block + "\n"
        else:
            print("Error: Could not find Pipeline class definition.")
            return

    target_file.write_text(new_content, encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/pipeline.py")
    append_methods(
        target_file,
        "Pipeline",
        "".join(format_builtin_method_signatures())
        + "".join(format_custom_method_signatures(Apply, Const, Echo, Parrot, Side)),
    )
