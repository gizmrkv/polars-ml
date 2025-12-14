from __future__ import annotations

import inspect
import re
from inspect import Parameter
from pathlib import Path
from typing import Any, Callable

import inflection
import polars as pl
from polars import DataFrame

from polars_ml.pipeline.basic import Apply, Const, Echo, Parrot, Side
from polars_ml.preprocessing import (
    BoxCoxTransform,
    Discretize,
    LabelEncode,
    LabelEncodeInverse,
    MinMaxScale,
    PowerTransformInverse,
    RobustScale,
    ScaleInverse,
    StandardScale,
    YeoJohnsonTransform,
)

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


def format_call_argument(param: inspect.Parameter) -> str:
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


def format_call_args(func: Callable[..., Any]) -> list[str]:
    sig = inspect.signature(func)
    return [format_call_argument(p) for p in sig.parameters.values()]


def format_pipe_getattr_method(name: str) -> str:
    method = getattr(DataFrame, name)
    sig = inspect.signature(method)
    return """
    def {name}({params}) -> Self:
        return self.pipe(GetAttr({call_args}))
    """.format(
        name=name,
        params=", ".join(
            format_param(p).replace(": DataFrame", ": DataFrame | Transformer")
            for p in sig.parameters.values()
        ),
        call_args=", ".join([f'"{name}"'] + format_call_args(method)[1:]),
    )


def format_pipe_transformer_method(transformer_cls: type) -> str:
    method = getattr(transformer_cls, "__init__")
    params = inspect.signature(method).parameters.values()
    return """
    def {method}({params}) -> Self:
        return self.pipe({name}({call_args}))
    """.format(
        method=inflection.underscore(transformer_cls.__name__),
        name=transformer_cls.__name__,
        params=", ".join(format_param(p) for p in params),
        call_args=", ".join(format_call_args(method)[1:]),
    )


def format_pipe_transform_method_with_inverse(
    transformer_cls: type, inverse_cls: type
) -> str:
    method = getattr(transformer_cls, "__init__")
    sig = inspect.signature(method)
    name = transformer_cls.__name__
    name_inv = inverse_cls.__name__
    return """
    @overload
    def {method}({params}) -> Self: ...

    @overload
    def {method}({params}, inverse_mapping: Mapping[str, str] | None) -> {name_inv}Context: ...

    def {method}({params}, inverse_mapping: Mapping[str, str] | None = None) -> Self | {name_inv}Context:
        if inverse_mapping is None:
            return self.pipe({name}({call_args}))
        else:
            return {name_inv}Context(
                self, {name}({call_args}), inverse_mapping
            )
            """.format(
        method=inflection.underscore(name),
        name=name,
        name_inv=name_inv,
        params=", ".join(format_param(p) for p in sig.parameters.values()),
        call_args=", ".join(format_call_args(method)[1:]),
    )


def update_methods(
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
        if f"class {class_name}" in content:
            new_content = content.rstrip() + "\n\n" + generated_block + "\n"
        else:
            print("Error: Could not find Pipeline class definition.")
            return

    target_file.write_text(new_content, encoding="utf-8")
    print("Done.")


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


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/pipeline.py")
    update_methods(
        target_file,
        "Pipeline",
        "".join(format_pipe_getattr_method(m) for m in get_dataframe_methods())
        + "".join(
            format_pipe_transformer_method(t)
            for t in [Apply, Const, Echo, Parrot, Side, Discretize]
        )
        + "".join(
            format_pipe_transform_method_with_inverse(t, i)
            for t, i in [
                (MinMaxScale, ScaleInverse),
                (StandardScale, ScaleInverse),
                (RobustScale, ScaleInverse),
                (BoxCoxTransform, PowerTransformInverse),
                (YeoJohnsonTransform, PowerTransformInverse),
                (LabelEncode, LabelEncodeInverse),
            ]
        ),
    )
