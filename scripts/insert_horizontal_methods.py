from __future__ import annotations

import inspect
from pathlib import Path

import inflection
from const import (
    END_INSERTION_MARKER,
    HORIZONTAL_NAMESPACES,
    START_INSERTION_MARKER,
)
from utils import insert_between_markers, render_call_args, render_params_sig


def render_methods() -> list[str]:
    import polars_ml.preprocessing.horizontal as horizontal

    codes = []
    template = """
    def {name}({params}) -> Pipeline:
        return self.pipeline.pipe({cls_name}(expr, *more_expr, {call_args}))
"""
    for name, obj in inspect.getmembers(horizontal):
        if (
            not name.startswith("Horizontal")
            or name in {"HorizontalAgg", "HorizontalNameSpace"}
            or not inspect.isclass(obj)
        ):
            continue

        cls_name = name
        # Convert HorizontalSum to sum
        method_name = inflection.underscore(cls_name.replace("Horizontal", ""))
        init_method = getattr(obj, "__init__")

        # Skip expr and more_expr which are handled by the namespace
        params_sig = render_params_sig(
            init_method, skip_params={"expr", "more_expr", "self"}
        )
        params = ", ".join(
            [
                "self, expr: IntoExpr | Iterable[IntoExpr], *more_expr: IntoExpr | Iterable[IntoExpr]"
            ]
            + params_sig
        )
        call_args = ", ".join(
            render_call_args(init_method, skip_params={"expr", "more_expr", "self"})
        )

        codes.append(
            template.format(
                name=method_name,
                params=params,
                cls_name=cls_name,
                call_args=call_args,
            )
        )

    return codes


def insert_horizontal_methods():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    target_file = PROJECT_ROOT / Path("src/polars_ml/preprocessing/horizontal.py")
    for _, namespace in HORIZONTAL_NAMESPACES:
        codes = render_methods()
        insert_between_markers(
            target_file,
            "".join(codes),
            START_INSERTION_MARKER.format(
                prefix=" " * 4, suffix=f" IN {namespace.__name__}"
            ),
            END_INSERTION_MARKER.format(
                prefix=" " * 4, suffix=f" IN {namespace.__name__}"
            ),
        )


if __name__ == "__main__":
    insert_horizontal_methods()
