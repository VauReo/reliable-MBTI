from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_val_test(
    df: pd.DataFrame,
    label_col: str,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / val / test with per-label stratification."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f'train_ratio + val_ratio + test_ratio must sum to 1.0, got {total!r}'
        )
    y = df[label_col]
    holdout_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=holdout_ratio,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )
    y_temp = temp_df[label_col]
    test_fraction_of_holdout = test_ratio / holdout_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_fraction_of_holdout,
        stratify=y_temp,
        random_state=random_state,
        shuffle=True,
    )
    return train_df, val_df, test_df
