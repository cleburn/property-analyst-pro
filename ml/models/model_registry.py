"""
Model Registry for Per-Metro Models

This module handles loading, saving, and versioning of trained models.
"""

import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np


class ModelRegistry:
    """
    Registry for managing per-metro models and a global fallback.

    Directory structure:
        artifacts_dir/
            global/
                model.joblib
                metadata.json
            per_metro/
                austin/
                    model.joblib
                    metadata.json
                dallas/
                    ...
    """

    def __init__(self, artifacts_dir: str = "ml/artifacts/v2"):
        """
        Initialize the model registry.

        Args:
            artifacts_dir: Base directory for model artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self._models_cache: Dict[str, Any] = {}

    def save_metro_model(
        self,
        metro: str,
        model: Any,
        imputer: Any,
        scaler: Any,
        feature_names: List[str],
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a trained per-metro model.

        Args:
            metro: Metro identifier (e.g., 'austin', 'dallas')
            model: Trained model object
            imputer: Fitted imputer for missing values
            scaler: Fitted scaler (or None if not used)
            feature_names: List of feature names
            metrics: Dict of evaluation metrics
            metadata: Additional metadata

        Returns:
            Path to saved model directory
        """
        metro_dir = self.artifacts_dir / "per_metro" / metro
        metro_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, metro_dir / "model.joblib")

        # Save preprocessing
        joblib.dump(imputer, metro_dir / "imputer.joblib")
        if scaler is not None:
            joblib.dump(scaler, metro_dir / "scaler.joblib")

        # Save feature names
        with open(metro_dir / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)

        # Build metadata
        full_metadata = {
            "metro": metro,
            "created_at": datetime.now().isoformat(),
            "model_type": type(model).__name__,
            "n_features": len(feature_names),
            "metrics": metrics,
            "has_scaler": scaler is not None,
            **(metadata or {})
        }

        with open(metro_dir / "metadata.json", 'w') as f:
            json.dump(full_metadata, f, indent=2)

        return metro_dir

    def save_global_model(
        self,
        model: Any,
        imputer: Any,
        encoder: Any,
        feature_names: List[str],
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save the global fallback model.

        Args:
            model: Trained model object
            imputer: Fitted imputer
            encoder: Fitted metro encoder
            feature_names: List of feature names
            metrics: Dict of evaluation metrics
            metadata: Additional metadata

        Returns:
            Path to saved model directory
        """
        global_dir = self.artifacts_dir / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, global_dir / "model.joblib")

        # Save preprocessing
        joblib.dump(imputer, global_dir / "imputer.joblib")
        joblib.dump(encoder, global_dir / "metro_encoder.joblib")

        # Save feature names
        with open(global_dir / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)

        # Build metadata
        full_metadata = {
            "model_type": "global",
            "created_at": datetime.now().isoformat(),
            "model_class": type(model).__name__,
            "n_features": len(feature_names),
            "metrics": metrics,
            **(metadata or {})
        }

        with open(global_dir / "metadata.json", 'w') as f:
            json.dump(full_metadata, f, indent=2)

        return global_dir

    def load_metro_model(self, metro: str) -> Optional[Dict[str, Any]]:
        """
        Load a per-metro model.

        Args:
            metro: Metro identifier

        Returns:
            Dict with model, imputer, scaler, feature_names, metadata
            or None if not found
        """
        # Check cache first
        cache_key = f"metro_{metro}"
        if cache_key in self._models_cache:
            return self._models_cache[cache_key]

        metro_dir = self.artifacts_dir / "per_metro" / metro

        if not metro_dir.exists():
            return None

        try:
            result = {
                "model": joblib.load(metro_dir / "model.joblib"),
                "imputer": joblib.load(metro_dir / "imputer.joblib"),
                "scaler": None,
                "feature_names": None,
                "metadata": None
            }

            # Load optional scaler
            scaler_path = metro_dir / "scaler.joblib"
            if scaler_path.exists():
                result["scaler"] = joblib.load(scaler_path)

            # Load feature names
            with open(metro_dir / "feature_names.json") as f:
                result["feature_names"] = json.load(f)

            # Load metadata
            with open(metro_dir / "metadata.json") as f:
                result["metadata"] = json.load(f)

            # Cache it
            self._models_cache[cache_key] = result
            return result

        except Exception as e:
            print(f"Error loading metro model for {metro}: {e}")
            return None

    def load_global_model(self) -> Optional[Dict[str, Any]]:
        """
        Load the global fallback model.

        Returns:
            Dict with model, imputer, encoder, feature_names, metadata
            or None if not found
        """
        # Check cache first
        cache_key = "global"
        if cache_key in self._models_cache:
            return self._models_cache[cache_key]

        global_dir = self.artifacts_dir / "global"

        if not global_dir.exists():
            return None

        try:
            result = {
                "model": joblib.load(global_dir / "model.joblib"),
                "imputer": joblib.load(global_dir / "imputer.joblib"),
                "encoder": joblib.load(global_dir / "metro_encoder.joblib"),
                "feature_names": None,
                "metadata": None
            }

            # Load feature names
            with open(global_dir / "feature_names.json") as f:
                result["feature_names"] = json.load(f)

            # Load metadata
            with open(global_dir / "metadata.json") as f:
                result["metadata"] = json.load(f)

            # Cache it
            self._models_cache[cache_key] = result
            return result

        except Exception as e:
            print(f"Error loading global model: {e}")
            return None

    def get_available_metros(self) -> List[str]:
        """
        Get list of metros with trained models.

        Returns:
            List of metro identifiers
        """
        per_metro_dir = self.artifacts_dir / "per_metro"
        if not per_metro_dir.exists():
            return []

        metros = []
        for metro_dir in per_metro_dir.iterdir():
            if metro_dir.is_dir() and (metro_dir / "model.joblib").exists():
                metros.append(metro_dir.name)

        return sorted(metros)

    def get_model_for_prediction(self, metro: str) -> Dict[str, Any]:
        """
        Get the best model for a given metro.

        Tries per-metro model first, falls back to global.

        Args:
            metro: Metro identifier

        Returns:
            Dict with model artifacts

        Raises:
            ValueError if no model is available
        """
        # Try per-metro first
        metro_model = self.load_metro_model(metro)
        if metro_model is not None:
            metro_model["source"] = f"per_metro/{metro}"
            return metro_model

        # Fall back to global
        global_model = self.load_global_model()
        if global_model is not None:
            global_model["source"] = "global"
            return global_model

        raise ValueError(f"No model available for metro '{metro}' and no global fallback")

    def clear_cache(self):
        """Clear the model cache."""
        self._models_cache.clear()
