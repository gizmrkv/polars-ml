from __future__ import annotations

from typing import Any, Self, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer


class MulticlassClassificationMetrics(Transformer):
    def __init__(
        self,
        y_true: str,
        y_probs: Sequence[str],
        *,
        by: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        top_k: int | Sequence[int] | None = None,
    ):
        self._y_true = y_true
        self._y_probs = y_probs
        self._by = by
        self._top_k = top_k

        self._train_class_dist: NDArray[Any] | None = None

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        # Calculate training class distribution for normalized entropy
        counts = data[self._y_true].value_counts()
        n_samples = len(data)
        n_classes = len(self._y_probs)

        dist = np.zeros(n_classes)
        for row in counts.iter_rows():
            label, count = row
            if 0 <= label < n_classes:
                dist[label] = count / n_samples

        self._train_class_dist = dist
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
                    by_data.select(self._y_probs).to_numpy(),
                )
                metrics_list.append(metrics)
        else:
            metrics = self.calc_metrics(
                data[self._y_true].to_numpy(),
                data.select(self._y_probs).to_numpy(),
            )
            metrics_list.append(metrics)

        return DataFrame(metrics_list)

    def calc_metrics(
        self, y_true: NDArray[Any], y_probs: NDArray[Any]
    ) -> dict[str, Any]:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            cohen_kappa_score,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
            top_k_accuracy_score,
        )

        y_pred = np.argmax(y_probs, axis=1)
        labels = np.arange(y_probs.shape[1])

        # Core metrics
        log_loss_value = log_loss(y_true, y_probs, labels=labels)
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        metrics = {
            "log_loss": log_loss_value,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "kappa": kappa,
        }

        # Normalized Entropy
        if self._train_class_dist is not None:
            # Base entropy: -sum(p * log(p))
            # Filter out zeros to avoid log(0)
            p = self._train_class_dist[self._train_class_dist > 0]
            base_entropy = -np.sum(p * np.log(p))
            if base_entropy > 0:
                metrics["normalized_entropy"] = log_loss_value / base_entropy
            else:
                metrics["normalized_entropy"] = None

        # Precision, Recall, F1
        for avg in ["macro", "weighted"]:
            metrics[f"precision_{avg}"] = precision_score(
                y_true, y_pred, average=avg, labels=labels, zero_division=0
            )
            metrics[f"recall_{avg}"] = recall_score(
                y_true, y_pred, average=avg, labels=labels, zero_division=0
            )
            metrics[f"f1_{avg}"] = f1_score(
                y_true, y_pred, average=avg, labels=labels, zero_division=0
            )

        # ROC AUC
        # roc_auc_score for multiclass requires at least one sample per class in y_true for some cases,
        # but OvR and OvO have different requirements.
        # We wrap in try-except to handle cases with missing classes in a group.
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true, y_probs, multi_class="ovr", labels=labels
            )
        except ValueError:
            metrics["roc_auc_ovr"] = None

        try:
            metrics["roc_auc_ovo"] = roc_auc_score(
                y_true, y_probs, multi_class="ovo", labels=labels
            )
        except ValueError:
            metrics["roc_auc_ovo"] = None

        # Top-k Accuracy
        if self._top_k is not None:
            top_ks = [self._top_k] if isinstance(self._top_k, int) else self._top_k
            for k in top_ks:
                if k <= 0 or k >= len(labels):
                    continue
                try:
                    metrics[f"accuracy@{k}"] = top_k_accuracy_score(
                        y_true, y_probs, k=k, labels=labels
                    )
                except ValueError:
                    metrics[f"accuracy@{k}"] = None

        return metrics
