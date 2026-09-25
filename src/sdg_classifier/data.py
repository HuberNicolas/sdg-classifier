"""Load the abstracts, split them into train/validation/test and plot the label distribution."""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

SDG_COLUMNS = [f"SDG{i:02d}" for i in range(1, 18)]


def load_abstracts(csv: Path) -> pd.DataFrame:
    """Read the output of abstract-scraper and keep the rows that have an abstract and at least one SDG label."""
    df = pd.read_csv(csv)
    missing = [c for c in ["abstract", *SDG_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv} is missing the columns {missing}")
    df = df[["abstract", *SDG_COLUMNS]].dropna(subset=["abstract"])
    df[SDG_COLUMNS] = df[SDG_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).clip(0, 1).astype(int)
    df = df[df[SDG_COLUMNS].sum(axis=1) > 0]
    return df.reset_index(drop=True)


def one_label_per_row(df: pd.DataFrame) -> pd.DataFrame:
    """Turn every article with n SDG labels into n rows with one label each."""
    long = df.melt(id_vars="abstract", value_vars=SDG_COLUMNS, var_name="sdg", value_name="on")
    long = long[long["on"] == 1]
    out = pd.get_dummies(long["sdg"]).reindex(columns=SDG_COLUMNS, fill_value=False).astype(int)
    out.insert(0, "abstract", long["abstract"].to_numpy())
    return out.reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = 42, test_size: float = 0.2, val_size: float = 0.2) -> DatasetDict:
    """Split into train, validation and test sets that keep the label proportions (iterative stratification)."""
    y = df[SDG_COLUMNS].to_numpy()
    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=seed)
    train_idx, rest_idx = next(first.split(np.zeros(len(y)), y))
    second = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size / (test_size + val_size), random_state=seed
    )
    val_rel, test_rel = next(second.split(np.zeros(len(rest_idx)), y[rest_idx]))
    parts = {"train": train_idx, "validation": rest_idx[val_rel], "test": rest_idx[test_rel]}
    return DatasetDict({name: Dataset.from_pandas(df.iloc[idx].reset_index(drop=True)) for name, idx in parts.items()})


def plot_distribution(df: pd.DataFrame, path: Path, title: str = "Distribution of SDG labels") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    df[SDG_COLUMNS].sum().plot(kind="bar", color="skyblue", edgecolor="black", ax=ax)
    ax.set(title=title, xlabel="SDG", ylabel="Articles")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
