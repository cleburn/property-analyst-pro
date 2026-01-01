"""
Expanding Window Cross-Validation for Time Series Data

This module provides cross-validation utilities that respect temporal ordering,
preventing data leakage from future to past.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Generator


class ExpandingWindowCV:
    """
    Expanding window cross-validation for temporal data.

    Unlike standard k-fold CV, this ensures training always uses earlier
    time periods to predict later ones, matching real-world deployment.

    Example:
        cv = ExpandingWindowCV(min_train_years=3)
        for train_idx, val_idx in cv.split(df, year_col='prediction_year'):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
    """

    def __init__(self, min_train_years: int = 3, step_size: int = 1):
        """
        Initialize expanding window CV.

        Args:
            min_train_years: Minimum number of years to include in first training set
            step_size: Number of years to add between folds (default: 1)
        """
        self.min_train_years = min_train_years
        self.step_size = step_size

    def split(self, df: pd.DataFrame, year_col: str = 'prediction_year') -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/validation indices for each fold.

        Args:
            df: DataFrame with a year column
            year_col: Name of the column containing years

        Yields:
            Tuple of (train_indices, validation_indices) as numpy arrays
        """
        years = sorted(df[year_col].unique())
        n_years = len(years)

        if n_years <= self.min_train_years:
            raise ValueError(f"Not enough years ({n_years}) for min_train_years={self.min_train_years}")

        # Generate folds
        for val_idx in range(self.min_train_years, n_years, self.step_size):
            train_years = years[:val_idx]
            val_year = years[val_idx]

            train_mask = df[year_col].isin(train_years)
            val_mask = df[year_col] == val_year

            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]

            yield train_indices, val_indices

    def get_n_splits(self, df: pd.DataFrame, year_col: str = 'prediction_year') -> int:
        """
        Get the number of CV folds.

        Args:
            df: DataFrame with a year column
            year_col: Name of the column containing years

        Returns:
            Number of folds
        """
        years = sorted(df[year_col].unique())
        n_years = len(years)

        if n_years <= self.min_train_years:
            return 0

        return (n_years - self.min_train_years - 1) // self.step_size + 1

    def get_fold_info(self, df: pd.DataFrame, year_col: str = 'prediction_year') -> List[dict]:
        """
        Get detailed information about each fold.

        Args:
            df: DataFrame with a year column
            year_col: Name of the column containing years

        Returns:
            List of dicts with fold information
        """
        years = sorted(df[year_col].unique())
        fold_info = []

        for i, (train_idx, val_idx) in enumerate(self.split(df, year_col)):
            train_years = df.iloc[train_idx][year_col].unique()
            val_years = df.iloc[val_idx][year_col].unique()

            fold_info.append({
                'fold': i + 1,
                'train_years': sorted(train_years),
                'val_years': sorted(val_years),
                'n_train': len(train_idx),
                'n_val': len(val_idx)
            })

        return fold_info
