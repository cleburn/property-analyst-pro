"""
Appreciation Predictor for Streamlit App Integration (v3)

This module provides ML-predicted appreciation rates for the investment analysis app.

Model C (24-month lookback):
- RandomForest price prediction model
- Trained on 2002-2024 data (36,681 examples)
- Forward validated: 2.63% MAPE on November 2025
- 24-month minimum history requirement
- 14 features (cagr_2yr instead of cagr_3yr/cagr_5yr)

Appreciation is derived from price predictions:
  appreciation_rate = (predicted_price - current_price) / current_price * 100

Predictions are pre-computed and stored in appreciation_predictions_current.csv
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List


class AppreciationPredictor:
    """
    Provides ML-predicted appreciation rates for neighborhoods.

    Uses Model C (24-month lookback) price predictions to derive appreciation rates.
    Forward validated: 2.63% MAPE on November 2025 data.

    Usage:
        predictor = get_predictor()
        if predictor:
            rates = predictor.get_predicted_appreciation_by_name()
            df['ml_appreciation'] = df['key'].map(rates)
    """

    def __init__(
        self,
        predictions_path: str = "data/processed/appreciation_predictions_current.csv",
        historical_path: str = "data/processed/ml_data_v3.pkl"
    ):
        """
        Initialize the predictor.

        Args:
            predictions_path: Path to direct appreciation model predictions
            historical_path: Path to historical CAGR data (fallback)
        """
        self.predictions_path = Path(predictions_path)
        self.historical_path = Path(historical_path)

        self._predictions_data: Optional[pd.DataFrame] = None
        self._historical_data: Optional[pd.DataFrame] = None
        self._name_to_prediction: Dict[str, float] = {}
        self._name_to_historical: Dict[str, float] = {}
        self._loaded = False

    def _load_data(self) -> bool:
        """Load appreciation predictions and historical data."""
        if self._loaded:
            return self._predictions_data is not None or self._historical_data is not None

        # Load direct appreciation model predictions (primary)
        if self.predictions_path.exists():
            try:
                self._predictions_data = pd.read_csv(self.predictions_path)

                # Create name+metro lookup (use display_metro to match app data)
                for _, row in self._predictions_data.iterrows():
                    key = f"{row['neighborhood_name']}_{row['display_metro']}"
                    self._name_to_prediction[key] = float(row['predicted_1yr_appreciation'])

                print(f"Loaded {len(self._name_to_prediction)} appreciation predictions")
            except Exception as e:
                print(f"Error loading appreciation predictions: {e}")

        # Load historical CAGRs (fallback)
        if self.historical_path.exists():
            try:
                with open(self.historical_path, 'rb') as f:
                    data = pickle.load(f)

                train_df = data['train_df']
                test_df = data['test_df']
                all_df = pd.concat([train_df, test_df], ignore_index=True)

                # Extract 2025 examples (most recent features)
                df_2025 = all_df[all_df['year'] == 2025].copy()

                if len(df_2025) > 0:
                    self._historical_data = df_2025

                    # Create name+metro lookup for historical CAGRs
                    for _, row in df_2025.iterrows():
                        key = f"{row['neighborhood_name']}_{row['metro']}"
                        if pd.notna(row.get('cagr_5yr')):
                            self._name_to_historical[key] = float(row['cagr_5yr'])

                    print(f"Loaded {len(self._name_to_historical)} historical CAGRs (fallback)")
            except Exception as e:
                print(f"Error loading historical data: {e}")

        self._loaded = True
        return len(self._name_to_prediction) > 0 or len(self._name_to_historical) > 0

    def is_available(self) -> bool:
        """Check if appreciation data is available."""
        self._load_data()
        return len(self._name_to_prediction) > 0 or len(self._name_to_historical) > 0

    def has_direct_predictions(self) -> bool:
        """Check if direct appreciation model predictions are available."""
        self._load_data()
        return len(self._name_to_prediction) > 0

    def get_available_metros(self) -> List[str]:
        """Get list of metros with appreciation data."""
        if not self.is_available():
            return []

        if self._predictions_data is not None:
            return sorted(self._predictions_data['display_metro'].unique().tolist())
        elif self._historical_data is not None:
            return sorted(self._historical_data['metro'].unique().tolist())
        return []

    def get_predicted_appreciation_by_name(self) -> Dict[str, float]:
        """
        Get ML-predicted appreciation rates keyed by 'name_metro'.

        Returns direct appreciation model predictions (preferred) with
        historical CAGR fallback for missing neighborhoods.

        Returns:
            Dict mapping 'neighborhood_metro' to appreciation rate (%)
        """
        if not self.is_available():
            return {}

        # Start with historical CAGRs
        rates = self._name_to_historical.copy()

        # Override with direct predictions where available
        rates.update(self._name_to_prediction)

        return rates

    def get_appreciation_rates_by_name(self, rate_type: str = 'predicted') -> Dict[str, float]:
        """
        Get appreciation rates keyed by 'name_metro'.

        Args:
            rate_type: 'predicted' for direct model, 'cagr_5yr' for historical

        Returns:
            Dict mapping 'neighborhood_metro' to appreciation rate
        """
        if not self.is_available():
            return {}

        if rate_type == 'predicted':
            return self.get_predicted_appreciation_by_name()
        else:
            # Return historical CAGRs only
            return self._name_to_historical.copy()

    def get_appreciation_by_name(
        self,
        neighborhood_name: str,
        metro: str,
        use_predicted: bool = True
    ) -> Optional[float]:
        """
        Get appreciation rate for a single neighborhood.

        Args:
            neighborhood_name: Neighborhood name
            metro: Metro identifier
            use_predicted: If True, prefer direct model predictions

        Returns:
            Appreciation rate as percentage, or None if not found
        """
        if not self.is_available():
            return None

        key = f"{neighborhood_name}_{metro}"

        if use_predicted and key in self._name_to_prediction:
            return self._name_to_prediction[key]

        if key in self._name_to_historical:
            return self._name_to_historical[key]

        return None

    def get_model_info(self) -> Dict:
        """Get information about the loaded models."""
        self._load_data()

        return {
            'has_direct_predictions': len(self._name_to_prediction) > 0,
            'num_direct_predictions': len(self._name_to_prediction),
            'has_historical_fallback': len(self._name_to_historical) > 0,
            'num_historical': len(self._name_to_historical),
            'metros': self.get_available_metros(),
            'model_type': 'Model C (24mo lookback, 2.63% MAPE)' if self.has_direct_predictions() else 'Historical CAGR'
        }

    def get_summary_stats(self) -> Dict:
        """Get summary statistics about the appreciation data."""
        if not self.is_available():
            return {}

        rates = self.get_predicted_appreciation_by_name()
        values = list(rates.values())

        if not values:
            return {}

        return {
            'total_neighborhoods': len(values),
            'metros': self.get_available_metros(),
            'mean_appreciation': float(np.mean(values)),
            'median_appreciation': float(np.median(values)),
            'std_appreciation': float(np.std(values)),
            'min_appreciation': float(np.min(values)),
            'max_appreciation': float(np.max(values)),
            'model_type': 'Model C (24mo lookback, 2.63% MAPE)' if self.has_direct_predictions() else 'Historical CAGR'
        }


def get_predictor(
    predictions_path: str = "data/processed/appreciation_predictions_current.csv",
    historical_path: str = "data/processed/ml_data_v3.pkl"
) -> Optional[AppreciationPredictor]:
    """
    Factory function to get a predictor if data is available.

    Returns:
        AppreciationPredictor instance if data exists, None otherwise
    """
    predictor = AppreciationPredictor(predictions_path, historical_path)

    if predictor.is_available():
        return predictor

    return None
