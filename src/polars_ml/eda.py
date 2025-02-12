import itertools
import shutil
from pathlib import Path
from typing import Literal

import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib import pyplot as plt
from polars import DataFrame
from tqdm import tqdm

from polars_ml.plot import iter_plots


class EDA:
    def __init__(
        self, data: DataFrame, save_dir: str | Path, *, show_progress: bool = True
    ):
        self.data = data
        self.save_dir = Path(save_dir)
        self.show_progress = show_progress

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def infer_numerical(self) -> list[str]:
        return self.data.lazy().select(cs.numeric()).collect_schema().names()

    def infer_categorical(self, threshold: int | float = 100) -> list[str]:
        is_ok = self.data.select(
            (
                pl.all().n_unique()
                < (threshold if isinstance(threshold, int) else threshold * pl.len())
            )
        ).row(0)
        return [col for col, is_cat in zip(self.data.columns, is_ok) if is_cat]

    def infer_constant(self, include_null: bool = False) -> list[str]:
        if include_null:
            is_ok = self.data.select(
                (
                    (pl.all().n_unique() == 1)
                    | ((pl.all().null_count() > 0) & (pl.all().n_unique() == 2))
                )
            ).row(0)
        else:
            is_ok = self.data.select(pl.all().n_unique() == 1).row(0)

        return [col for col, is_const in zip(self.data.columns, is_ok) if is_const]

    def infer_high_null(self, threshold: float = 0.75) -> list[str]:
        is_ok = self.data.select(pl.all().null_count() > threshold * pl.len()).row(0)
        return [
            col for col, is_high_null in zip(self.data.columns, is_ok) if is_high_null
        ]

    def infer_probability(self) -> list[str]:
        is_ok = self.data.select(
            pl.col(c).is_between(0, 1) for c in self.infer_numerical()
        ).row(0)
        return [col for col, is_prob in zip(self.data.columns, is_ok) if is_prob]

    def describe(self):
        self.data.describe().write_csv(self.save_dir / "describe.csv")

    def categorical_count(self, threshold: int | float = 100):
        count = (
            self.data.select(self.infer_categorical(threshold))
            .unpivot(cs.all())
            .group_by("variable", "value")
            .agg(pl.len().alias("count"))
            .sort("variable", "value")
        )
        count.write_csv(self.save_dir / "categorical_count.csv")

    def plot_corr_heatmap(
        self,
        *,
        figsize: tuple[int, int] = (10, 10),
        annot: bool = False,
        square: bool = False,
    ):
        data = self.data.select(self.infer_numerical())
        m = data.corr().to_numpy()
        labels = data.columns
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            m,
            annot=annot,
            square=square,
            xticklabels=labels,
            yticklabels=labels,
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            ax=ax,
        )
        fig.savefig(self.save_dir / "corr_heatmap.png")
        fig.clear()
        plt.close(fig)

    def plot_count_heatmap(
        self,
        *,
        categorical_threshold: int | float = 100,
        figsize: tuple[int, int] = (10, 10),
        annot: bool = False,
        square: bool = False,
    ):
        save_dir = self.save_dir / "count_heatmap"
        shutil.rmtree(save_dir, ignore_errors=True)
        save_dir.mkdir(exist_ok=True)
        cats = self.infer_categorical(threshold=categorical_threshold)
        for cat1, cat2 in tqdm(
            itertools.combinations(cats, 2),
            total=len(cats) * (len(cats) - 1) // 2,
            disable=not self.show_progress,
        ):
            m = (
                self.data.group_by(
                    pl.col(cat1).cast(pl.String),
                    pl.col(cat2).cast(pl.String),
                )
                .len()
                .fill_null("")
                .sort(pl.all())
                .pivot(
                    cat2,
                    index=cat1,
                    values="len",
                    sort_columns=True,
                    maintain_order=True,
                )
                .fill_null(0)
            )
            x_labels = m.columns[1:]
            y_labels = m[cat1].to_list()

            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                m.drop(cat1).to_numpy(),
                annot=annot,
                square=square,
                fmt="d",
                ax=ax,
                cmap="viridis",
            )
            ax.set_xticklabels(x_labels, rotation=90)
            ax.set_yticklabels(y_labels, rotation=0)
            ax.set_xlabel(cat2)
            ax.set_ylabel(cat1)
            ax.set_title(f"{cat1} vs {cat2}")
            fig.savefig(save_dir / f"{cat1} vs {cat2}.png")
            fig.clear()
            plt.close(fig)

    def plot_histogram(
        self,
        *,
        categorical_threshold: int | float = 100,
        figsize: tuple[int, int] = (12, 8),
        bins: int | Literal["auto"] = "auto",
    ):
        save_dir = self.save_dir / "histogram"
        shutil.rmtree(save_dir, ignore_errors=True)
        save_dir.mkdir(exist_ok=True)
        xs = self.infer_numerical()
        hues = [None] + self.infer_categorical(categorical_threshold)
        for x, hue in tqdm(
            itertools.product(xs, hues),
            total=len(xs) * len(hues),
            disable=not self.show_progress,
        ):
            fig, ax = plt.subplots(figsize=figsize)
            sns.histplot(
                data=self.data.select(set([x, hue]) if hue else x),
                x=x,
                hue=hue,
                bins=bins,
                kde=True,
                alpha=0.5,
                element="step",
                ax=ax,
            )
            filename = x + (f" by {hue}" if hue else "")
            fig.savefig(save_dir / f"{filename}.png")
            fig.clear()
            plt.close(fig)

    def plot_scatter(
        self,
        *,
        categorical_threshold: int | float = 100,
        figsize: tuple[int, int] = (10, 10),
    ):
        save_dir = self.save_dir / "scatter"
        shutil.rmtree(save_dir, ignore_errors=True)
        save_dir.mkdir(exist_ok=True)
        nums = self.infer_numerical()
        hues = [None] + self.infer_categorical(categorical_threshold)
        x_y_hues = list(itertools.product(itertools.combinations(nums, 2), hues))
        for (x, y), hue in tqdm(
            x_y_hues, total=len(x_y_hues), disable=not self.show_progress
        ):
            fig, ax = plt.subplots(figsize=figsize)
            tmp = self.data.select(set([x, y, hue]) if hue else set([x, y]))
            sns.scatterplot(
                tmp, x=x, y=y, hue=hue, s=10, edgecolor=None, alpha=0.5, ax=ax
            )
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{x} vs {y}" + (f" by {hue}" if hue else ""))
            if hue:
                ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

            fig.tight_layout()

            filename = f"{x} vs {y}" + (f" by {hue}" if hue else "")
            fig.savefig(save_dir / f"{filename}.png")
            fig.clear()
            plt.close(fig)
