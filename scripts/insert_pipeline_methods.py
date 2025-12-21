from __future__ import annotations

import inspect
from pathlib import Path

import inflection
import polars as pl
from const import (
    BASIC_TRANSFORMERS,
    BASIC_TRANSFORMERS_WITH_INVERSE,
    BUILT_IN_FUNCTION_BLOCK_LIST,
    END_INSERTION_MARKER,
    GROUP_BY_NAMESPACES,
    START_INSERTION_MARKER,
)
from polars import DataFrame
from utils import insert_between_markers, render_call_args, render_params_sig


def render_methods() -> list[str]:
    codes = []

    template = """
    def {name}({params}) -> Self:
        return self.pipe({cls_name}({call_args}))
"""
    for name, obj in inspect.getmembers(DataFrame):
        if (
            name.startswith("_")
            or not callable(obj)
            or name in BUILT_IN_FUNCTION_BLOCK_LIST
        ):
            continue

        params = ", ".join(
            ["self"]
            + render_params_sig(
                obj,
                override_annotations={"DataFrame": "DataFrame | Transformer"},
            )
        )
        call_args = ", ".join([f'"{name}"'] + render_call_args(obj))

        if inspect.signature(obj).return_annotation in {"DataFrame", "Self"}:
            codes.append(
                template.format(
                    name=name, params=params, call_args=call_args, cls_name="GetAttr"
                )
            )

        if name.startswith("write_"):
            codes.append(
                template.format(
                    name=name, params=params, call_args=call_args, cls_name="GetAttr"
                )
            )

    for name, obj in inspect.getmembers(pl):
        if (
            not name.startswith("read_")
            or not callable(obj)
            or name in BUILT_IN_FUNCTION_BLOCK_LIST
        ):
            continue

        params = ", ".join(
            ["self"]
            + render_params_sig(
                obj,
                override_annotations={"DataFrame": "DataFrame | Transformer"},
            )
        )
        call_args = ", ".join([f'"{name}"'] + render_call_args(obj))
        codes.append(
            template.format(
                name=name,
                params=params,
                call_args=call_args,
                cls_name="GetAttrPolars",
            )
        )

    template = """
    def {name}({params}) -> {namespace}:
        return {namespace}(self, "{name}", {call_args})
"""
    for name, namespace in GROUP_BY_NAMESPACES:
        obj = getattr(DataFrame, name)
        params = ", ".join(
            ["self"]
            + render_params_sig(
                obj,
                override_annotations={"DataFrame": "DataFrame | Transformer"},
            )
        )
        namespace = namespace.__name__
        call_args = ", ".join(render_call_args(obj))
        codes.append(
            template.format(
                name=name, params=params, namespace=namespace, call_args=call_args
            )
        )

    template = """
    def {name}({params}) -> Self:
        return self.pipe({cls_name}({call_args}))
"""
    for transformer_cls in BASIC_TRANSFORMERS:
        obj = getattr(transformer_cls, "__init__")
        cls_name = transformer_cls.__name__
        name = inflection.underscore(cls_name)
        params = ", ".join(["self"] + render_params_sig(obj))
        call_args = ", ".join(render_call_args(obj))
        codes.append(
            template.format(
                name=name, params=params, cls_name=cls_name, call_args=call_args
            )
        )

    template = """
    @overload
    def {name}({params}) -> Self: ...

    @overload
    def {name}({params}, inverse_mapping: Mapping[str, str] | None) -> {context}: ...

    def {name}({params}, inverse_mapping: Mapping[str, str] | None = None) -> Self | {context}:
        if inverse_mapping is None:
            return self.pipe({cls_name}({call_args}))
        else:
            return {context}(
                self, {cls_name}({call_args}), inverse_mapping
            )
"""
    for transformer_cls, context_cls in BASIC_TRANSFORMERS_WITH_INVERSE:
        obj = getattr(transformer_cls, "__init__")
        cls_name = transformer_cls.__name__
        name = inflection.underscore(cls_name)
        context = context_cls.__name__
        params = ", ".join(["self"] + render_params_sig(obj))
        call_args = ", ".join(render_call_args(obj))
        codes.append(
            template.format(
                name=name,
                params=params,
                context=context,
                cls_name=cls_name,
                call_args=call_args,
            )
        )

    return codes


def insert_pipeline_methods():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/pipeline.py")
    codes = render_methods()
    insert_between_markers(
        target_file,
        "".join(codes),
        START_INSERTION_MARKER.format(prefix=" " * 4, suffix=" IN Pipeline"),
        END_INSERTION_MARKER.format(prefix=" " * 4, suffix=" IN Pipeline"),
    )


if __name__ == "__main__":
    insert_pipeline_methods()
