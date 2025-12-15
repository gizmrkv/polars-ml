from __future__ import annotations

import inspect
import re
from inspect import Parameter
from pathlib import Path
from typing import Any, Callable

import inflection
from polars import DataFrame
from polars.dataframe.group_by import DynamicGroupBy, GroupBy, RollingGroupBy

from polars_ml.gbdt import LightGBM, LightGBMTuner, LightGBMTunerCV, XGBoost
from polars_ml.metrics import BinaryClassificationMetrics, RegressionMetrics
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


def update_methods(target_file: str | Path, class_name: str, code: str) -> None:
    import textwrap
    from pathlib import Path

    path = Path(target_file)
    content = path.read_text(encoding="utf-8")

    lines = content.splitlines(True)  # keep line endings

    class_re = re.compile(rf"^(\s*)class\s+{re.escape(class_name)}\b.*:\s*(#.*)?$")
    cls_i = None
    cls_indent = 0
    for i, line in enumerate(lines):
        m = class_re.match(line.rstrip("\r\n"))
        if m:
            cls_i = i
            cls_indent = len(m.group(1))
            break
    if cls_i is None:
        raise ValueError(f"class {class_name} not found in {path}")

    body_indent = " " * (cls_indent + 4)
    start = f"{body_indent}# --- BEGIN AUTO-GENERATED METHODS IN {class_name} ---"
    end = f"{body_indent}# --- END AUTO-GENERATED METHODS IN {class_name} ---"

    body = textwrap.dedent(code).strip("\n")
    body_lines = [body_indent + ln if ln else "" for ln in body.splitlines()]
    block = [start + "\n", *[ln + "\n" for ln in body_lines], end + "\n"]

    def strip_nl(s: str) -> str:
        return s.rstrip("\r\n")

    s_i = None
    for i, line in enumerate(lines):
        if strip_nl(line) == start:
            s_i = i
            break
    if s_i is not None:
        e_i = None
        for j in range(s_i + 1, len(lines)):
            if strip_nl(lines[j]) == end:
                e_i = j
                break
        if e_i is None:
            raise ValueError(f"start marker found but end marker missing: {class_name}")
        lines[s_i : e_i + 1] = block
        path.write_text("".join(lines), encoding="utf-8")
        return

    insert_at = len(lines)
    for k in range(cls_i + 1, len(lines)):
        t = lines[k]
        if t.strip() == "":
            continue
        indent = len(t) - len(t.lstrip(" "))
        if indent <= cls_indent:
            insert_at = k
            break

    lines[insert_at:insert_at] = block
    path.write_text("".join(lines), encoding="utf-8")


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


def get_group_by_methods(
    group_by_type: type[GroupBy] | type[DynamicGroupBy] | type[RollingGroupBy],
) -> list[str]:
    methods: list[str] = []
    for name, obj in inspect.getmembers(group_by_type):
        if name.startswith("_") or not callable(obj):
            continue

        ret = inspect.signature(obj).return_annotation
        if ret in {"DataFrame"}:
            methods.append(name)

    return sorted(set(methods))


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/pipeline.py")
    template = """
    def {name}({params}) -> Self:
        return self.pipe(GetAttr({call_args}))
    """
    codes = []
    for name in get_dataframe_methods():
        method = getattr(DataFrame, name)
        sig = inspect.signature(method)
        codes.append(
            template.format(
                name=name,
                params=", ".join(
                    format_param(p).replace(": DataFrame", ": DataFrame | Transformer")
                    for p in sig.parameters.values()
                ),
                call_args=", ".join([f'"{name}"'] + format_call_args(method)[1:]),
            )
        )

    template = """
    def {name}({params}) -> GroupByNameSpace:
        return GroupByNameSpace(self, {call_args})
    """
    name = "group_by"
    method = getattr(DataFrame, name)
    sig = inspect.signature(method)
    codes.append(
        template.format(
            name=name,
            params=", ".join(format_param(p) for p in sig.parameters.values()),
            call_args=", ".join([f'"{name}"'] + format_call_args(method)[1:]),
        )
    )

    template = """
    def {method}({params}) -> Self:
        return self.pipe({name}({call_args}))
    """
    for transformer_cls in [Apply, Const, Echo, Parrot, Side, Discretize]:
        method = getattr(transformer_cls, "__init__")
        params = inspect.signature(method).parameters.values()
        name = transformer_cls.__name__
        method_name = inflection.underscore(name)
        codes.append(
            template.format(
                method=method_name,
                name=name,
                params=", ".join(format_param(p) for p in params),
                call_args=", ".join(format_call_args(method)[1:]),
            )
        )

    template = """
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
    """
    for transformer_cls, inverse_cls in [
        (MinMaxScale, ScaleInverse),
        (StandardScale, ScaleInverse),
        (RobustScale, ScaleInverse),
        (BoxCoxTransform, PowerTransformInverse),
        (YeoJohnsonTransform, PowerTransformInverse),
        (LabelEncode, LabelEncodeInverse),
    ]:
        method = getattr(transformer_cls, "__init__")
        sig = inspect.signature(method)
        name = transformer_cls.__name__
        name_inv = inverse_cls.__name__
        codes.append(
            template.format(
                method=inflection.underscore(name),
                name=name,
                name_inv=name_inv,
                params=", ".join(format_param(p) for p in sig.parameters.values()),
                call_args=", ".join(format_call_args(method)[1:]),
            )
        )

    update_methods(target_file, "Pipeline", "".join(codes))

    target_file = PROJECT_ROOT / Path("src/polars_ml/gbdt/__init__.py")
    codes = []
    for transformer_cls, method_name in [
        (LightGBM, "lightgbm"),
        (XGBoost, "xgboost"),
        (LightGBMTuner, "lightgbm_tuner"),
        (LightGBMTunerCV, "lightgbm_tuner_cv"),
    ]:
        method = getattr(transformer_cls, "__init__")
        params = inspect.signature(method).parameters.values()
        name = transformer_cls.__name__
        codes.append(
            """
    def {method}({params}) -> "Pipeline":
        return self.pipeline.pipe({name}({call_args}))
        """.format(
                method=method_name,
                name=name,
                params=", ".join(format_param(p) for p in params),
                call_args=", ".join(format_call_args(method)[1:]),
            )
        )

    update_methods(target_file, "GBDTNameSpace", "".join(codes))

    target_file = PROJECT_ROOT / Path("src/polars_ml/metrics/__init__.py")
    codes = []
    for transformer_cls, method_name in [
        (BinaryClassificationMetrics, "binary_classification"),
        (RegressionMetrics, "regression"),
    ]:
        method = getattr(transformer_cls, "__init__")
        params = inspect.signature(method).parameters.values()
        name = transformer_cls.__name__
        codes.append(
            """
    def {method}({params}) -> "Pipeline":
        return self.pipeline.pipe({name}({call_args}))
        """.format(
                method=method_name,
                name=name,
                params=", ".join(format_param(p) for p in params),
                call_args=", ".join(format_call_args(method)[1:]),
            )
        )

    update_methods(target_file, "MetricsNameSpace", "".join(codes))

    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/group_by.py")
    for group_by_type in [GroupBy, DynamicGroupBy, RollingGroupBy]:
        group_by_name = group_by_type.__name__
        codes = []
        template = """
    def {name}({params}) -> "Pipeline":
        return self.pipeline.pipe(GroupByGetAttr(self.attr, "{name}", self.args, self.kwargs, {call_args}))
        """
        codes = []
        for name in get_group_by_methods(group_by_type):
            method = getattr(group_by_type, name)
            sig = inspect.signature(method)
            codes.append(
                template.format(
                    name=name,
                    params=", ".join(format_param(p) for p in sig.parameters.values()),
                    call_args=", ".join(format_call_args(method)[1:]),
                )
            )

        update_methods(
            target_file,
            f"{group_by_name}NameSpace",
            "".join(codes),
        )
