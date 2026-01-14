from __future__ import annotations

import inspect
from pathlib import Path

import inflection
from const import (
    END_INSERTION_MARKER,
    FEATURE_ENGINEERING_NAMESPACES,
    START_INSERTION_MARKER,
)
from utils import insert_between_markers, render_call_args, render_params_sig


def render_methods() -> list[str]:
    import polars_ml.feature_engineering as fe

    codes = []
    template = """
    def {name}({params}) -> Pipeline:
        return self.pipeline.pipe({cls_name}({call_args}))
"""
    for name, obj in inspect.getmembers(fe):
        if name in {"FeatureEngineeringNameSpace"} or not inspect.isclass(obj):
            continue

        cls_name = name
        method_name = inflection.underscore(cls_name)
        init_method = getattr(obj, "__init__")

        params_sig = render_params_sig(init_method, skip_params={"self"})
        params = ", ".join(["self"] + params_sig)
        call_args = ", ".join(render_call_args(init_method, skip_params={"self"}))

        codes.append(
            template.format(
                name=method_name,
                params=params,
                cls_name=cls_name,
                call_args=call_args,
            )
        )

    return codes


def insert_feature_engineering_methods():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    target_file = PROJECT_ROOT / Path("src/polars_ml/feature_engineering/__init__.py")
    for _, namespace in FEATURE_ENGINEERING_NAMESPACES:
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
    insert_feature_engineering_methods()
