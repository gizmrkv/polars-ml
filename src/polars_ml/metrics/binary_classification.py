from __future__ import annotations

from typing import Any, Self, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer


class BinaryClassificationMetrics(Transformer):
    def __init__(
        self,
        y_true: str,
        y_prob: str,
        *,
        by: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        threshold: float | None = None,
        top_k: int | Sequence[int] | None = None,
    ):
        self._y_true = y_true
        self._y_prob = y_prob
        self._by = by
        self._threshold = threshold
        self._top_k = top_k

        self._train_pos_rate: float | None = None

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        self._train_pos_rate = data[self._y_true].sum() / len(data)  # type: ignore
        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        metrics_list: list[dict[str, Any]] = []
        if self._by is not None:
            by_columns = data.lazy().select(self._by).collect_schema().names()
            for by_value, by_data in data.partition_by(
                self._by, maintain_order=True, include_key=False, as_dict=True
            ).items():
                metrics = dict(zip(by_columns, by_value)) | self.calc_metrics(
                    by_data[self._y_true].to_numpy(),
                    by_data[self._y_prob].to_numpy(),
                )
                metrics_list.append(metrics)
        else:
            metrics = self.calc_metrics(
                data[self._y_true].to_numpy(),
                data[self._y_prob].to_numpy(),
            )
            metrics_list.append(metrics)

        return pl.DataFrame(metrics_list)

    def calc_metrics(
        self, y_true: NDArray[Any], y_pred: NDArray[Any]
    ) -> dict[str, Any]:
        from sklearn.metrics import (
            accuracy_score,
            auc,
            average_precision_score,
            brier_score_loss,
            confusion_matrix,
            f1_score,
            log_loss,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        n_positives = y_true.sum()
        has_positives = n_positives > 0

        log_loss_value = log_loss(y_true, y_pred)

        if self._train_pos_rate is not None:
            entropy = -(
                self._train_pos_rate * np.log(self._train_pos_rate)
                + (1 - self._train_pos_rate) * np.log(1 - self._train_pos_rate)
            )
            normalized_entropy = log_loss_value / entropy
        else:
            normalized_entropy = None

        if has_positives:
            roc_auc = roc_auc_score(y_true, y_pred)
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = auc(recall, precision)
            average_precision = average_precision_score(y_true, y_pred)
        else:
            roc_auc = None
            pr_auc = None
            average_precision = None

        brier_score = brier_score_loss(y_true, y_pred)

        metrics = {
            "log_loss": log_loss_value,
            "normalized_entropy": normalized_entropy,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "average_precision": average_precision,
            "brier_score": brier_score,
        }

        if self._threshold is not None:
            y_pred_binary = (y_pred >= self._threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            accuracy = accuracy_score(y_true, y_pred_binary)

            if has_positives:
                precision = precision_score(y_true, y_pred_binary, zero_division=0)
                recall = recall_score(y_true, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true, y_pred_binary, zero_division=0)
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            else:
                precision = None
                recall = None
                f1 = None
                fnr = None

            n_negatives = len(y_true) - n_positives
            if n_negatives > 0:
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            else:
                specificity = None
                npv = None
                fpr = None

            metrics.update(
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "specificity": specificity,
                    "npv": npv,
                    "fpr": fpr,
                    "fnr": fnr,
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                }
            )

        if self._top_k is not None:
            top_ks = [self._top_k] if isinstance(self._top_k, int) else self._top_k
            sort_indices = np.argsort(y_pred)[::-1]
            sorted_y_true = y_true[sort_indices]

            for k in top_ks:
                if k <= 0:
                    continue
                k_val = min(k, len(sorted_y_true))
                top_k_y_true = sorted_y_true[:k_val]
                n_tp_at_k = top_k_y_true.sum()

                metrics[f"precision@{k}"] = n_tp_at_k / k_val if k_val > 0 else 0.0
                metrics[f"recall@{k}"] = (
                    n_tp_at_k / n_positives if n_positives > 0 else None
                )

        return metrics
