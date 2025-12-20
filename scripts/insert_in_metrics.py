from __future__ import annotations

from pathlib import Path

from const import END_INSERTION_MARKER, METRICS_TRANSFORMERS, START_INSERTION_MARKER
from utils import insert_between_markers, render_call_args, render_params_sig


def render_methods() -> list[str]:
    codes = []
    template = """
    def {name}({params}) -> Pipeline:
        return self.pipeline.pipe({cls_name}({call_args}))
"""
    for method_name, transformer_cls in METRICS_TRANSFORMERS:
        obj = getattr(transformer_cls, "__init__")
        cls_name = transformer_cls.__name__
        params = ", ".join(["self"] + render_params_sig(obj))
        call_args = ", ".join(render_call_args(obj))
        codes.append(
            template.format(
                name=method_name, params=params, cls_name=cls_name, call_args=call_args
            )
        )

    return codes


def insert_in_metrics():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    target_file = PROJECT_ROOT / Path("src/polars_ml/metrics/__init__.py")
    codes = render_methods()
    insert_between_markers(
        target_file,
        "".join(codes),
        START_INSERTION_MARKER.format(prefix=" " * 4, suffix=" IN MetricsNameSpace"),
        END_INSERTION_MARKER.format(prefix=" " * 4, suffix=" IN MetricsNameSpace"),
    )


if __name__ == "__main__":
    insert_in_metrics()
