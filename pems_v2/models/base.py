"""
Base model infrastructure for PEMS v2.

Provides abstract base classes and utilities for all predictive models with:
- Unified interface for training, prediction, and evaluation
- Automatic versioning and metadata tracking
- Performance monitoring and drift detection
- Model persistence and deployment management
- Online learning capabilities
"""

import hashlib
import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelMetadata:
    """Comprehensive model metadata for versioning and tracking."""

    model_name: str
    version: str
    training_date: datetime
    features: List[str]
    target_variable: str
    performance_metrics: Dict[str, float]
    training_params: Dict[str, Any]
    data_hash: str
    model_type: str
    deployment_status: str = "development"
    training_samples: int = 0
    validation_samples: int = 0
    feature_importance: Optional[Dict[str, float]] = None
    model_size_mb: float = 0.0
    training_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["training_date"] = self.training_date.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        data["training_date"] = datetime.fromisoformat(data["training_date"])
        return cls(**data)


@dataclass
class PredictionResult:
    """Structured prediction result with uncertainty quantification."""

    predictions: Union[pd.Series, pd.DataFrame]
    uncertainty: Optional[Union[pd.Series, pd.DataFrame]] = None
    feature_contributions: Optional[pd.DataFrame] = None
    prediction_timestamp: datetime = None
    model_version: str = ""
    confidence_intervals: Optional[Dict[str, Union[pd.Series, pd.DataFrame]]] = None

    def __post_init__(self):
        if self.prediction_timestamp is None:
            self.prediction_timestamp = datetime.now()


@dataclass
class PerformanceMetrics:
    """Standardized performance metrics across all models."""

    mae: float
    rmse: float
    r2: float
    mape: float
    bias: float
    std_residual: float
    max_error: float
    explained_variance: float

    @classmethod
    def calculate(cls, y_true: np.ndarray, y_pred: np.ndarray) -> "PerformanceMetrics":
        """Calculate comprehensive performance metrics."""
        # Handle edge cases
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) == 0 or len(y_pred) == 0:
            return cls.empty()

        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Additional metrics
        residuals = y_true - y_pred
        bias = np.mean(residuals)
        std_residual = np.std(residuals)
        max_error = np.max(np.abs(residuals))

        # MAPE with protection against division by zero
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = (
                np.mean(np.abs(residuals[non_zero_mask] / y_true[non_zero_mask])) * 100
            )
        else:
            mape = float("inf")

        # Explained variance
        variance_y = np.var(y_true)
        explained_variance = (
            1 - (np.var(residuals) / variance_y) if variance_y > 0 else 0
        )

        return cls(
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            bias=bias,
            std_residual=std_residual,
            max_error=max_error,
            explained_variance=explained_variance,
        )

    @classmethod
    def empty(cls) -> "PerformanceMetrics":
        """Return empty metrics for error cases."""
        return cls(
            mae=float("inf"),
            rmse=float("inf"),
            r2=-float("inf"),
            mape=float("inf"),
            bias=0.0,
            std_residual=0.0,
            max_error=0.0,
            explained_variance=0.0,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


class BasePredictor(ABC):
    """Abstract base class for all PEMS predictive models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize base predictor with configuration."""
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.metadata: Optional[ModelMetadata] = None
        self.performance_history: List[PerformanceMetrics] = []

        # Logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Model paths
        self.model_dir = Path(config.get("model_dir", "models/saved"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Training configuration
        self.random_state = config.get("random_state", 42)
        self.validation_split = config.get("validation_split", 0.2)
        self.enable_online_learning = config.get("enable_online_learning", False)

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs,
    ) -> PerformanceMetrics:
        """
        Train the model on historical data.

        Args:
            X: Feature matrix
            y: Target variable
            validation_data: Optional validation set
            **kwargs: Additional training parameters

        Returns:
            Performance metrics on validation set
        """
        pass

    @abstractmethod
    def predict(
        self, X: pd.DataFrame, return_uncertainty: bool = False, **kwargs
    ) -> PredictionResult:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix
            return_uncertainty: Whether to include uncertainty estimates
            **kwargs: Additional prediction parameters

        Returns:
            Structured prediction result
        """
        pass

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training or prediction.

        Args:
            X: Raw feature matrix

        Returns:
            Processed feature matrix
        """
        # Ensure feature columns match training
        if self.feature_columns is not None:
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                self.logger.warning(f"Missing features: {missing_cols}")
                # Add missing columns with zeros
                for col in missing_cols:
                    X[col] = 0.0

            # Select and reorder columns
            X = X[self.feature_columns]

        # Apply scaling if trained
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X), index=X.index, columns=X.columns
            )
            return X_scaled

        return X

    def fit_preprocessing(self, X: pd.DataFrame) -> None:
        """
        Fit preprocessing steps (scaling, feature selection).

        Args:
            X: Training feature matrix
        """
        # Store feature columns
        self.feature_columns = list(X.columns)

        # Fit scaler
        if self.config.get("scale_features", True):
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            self.logger.info(f"Fitted scaler for {len(X.columns)} features")

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None
    ) -> PerformanceMetrics:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: True target values
            sample_weight: Optional sample weights

        Returns:
            Performance metrics
        """
        predictions = self.predict(X)

        if isinstance(predictions.predictions, pd.Series):
            y_pred = predictions.predictions.values
        else:
            y_pred = predictions.predictions.iloc[:, 0].values

        y_true = y.values

        # Apply sample weights if provided
        if sample_weight is not None:
            # Weighted metrics calculation would go here
            pass

        metrics = PerformanceMetrics.calculate(y_true, y_pred)

        # Add to performance history
        self.performance_history.append(metrics)

        return metrics

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.

        Returns:
            Dictionary of feature names to importance scores
        """
        if self.model is None or self.feature_columns is None:
            return None

        # Try to extract feature importance from model
        importance = None

        if hasattr(self.model, "feature_importances_"):
            # Tree-based models
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # Linear models
            importance = np.abs(self.model.coef_)
        elif hasattr(self.model, "feature_importance_"):
            # LightGBM/XGBoost
            importance = self.model.feature_importance_()

        if importance is not None:
            return dict(zip(self.feature_columns, importance))

        return None

    def calculate_data_hash(self, X: pd.DataFrame, y: pd.Series) -> str:
        """
        Calculate hash of training data for versioning.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            SHA256 hash of the data
        """
        # Combine features and target
        data_str = f"{X.to_string()}{y.to_string()}"
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def generate_version(self, training_data_hash: str) -> str:
        """
        Generate model version string.

        Args:
            training_data_hash: Hash of training data

        Returns:
            Version string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = self.__class__.__name__
        return f"{model_type}_{timestamp}_{training_data_hash[:8]}"

    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        Save trained model with metadata.

        Args:
            path: Optional custom save path

        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No trained model to save")

        if path is None:
            path = (
                self.model_dir
                / f"{self.metadata.model_name}_{self.metadata.version}.pkl"
            )

        # Save model components
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "metadata": self.metadata,
            "config": self.config,
            "performance_history": self.performance_history,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        # Save metadata separately for easy access
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        # Calculate model size
        model_size_mb = path.stat().st_size / (1024 * 1024)
        self.metadata.model_size_mb = model_size_mb

        self.logger.info(f"Model saved to {path} ({model_size_mb:.2f} MB)")
        return path

    def load_model(self, path: Path) -> None:
        """
        Load trained model with metadata.

        Args:
            path: Path to saved model
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.metadata = model_data["metadata"]
        self.performance_history = model_data.get("performance_history", [])

        # Update config with saved config
        saved_config = model_data.get("config", {})
        self.config.update(saved_config)

        self.logger.info(f"Model loaded from {path}")

    def update_online(
        self, X_new: pd.DataFrame, y_new: pd.Series, learning_rate: float = 0.01
    ) -> bool:
        """
        Update model with new data (online learning).

        Args:
            X_new: New feature data
            y_new: New target data
            learning_rate: Learning rate for update

        Returns:
            True if update was successful
        """
        if not self.enable_online_learning:
            self.logger.warning("Online learning not enabled for this model")
            return False

        if self.model is None:
            self.logger.error("No trained model to update")
            return False

        try:
            # Prepare features
            X_processed = self.prepare_features(X_new)

            # Update model (implementation depends on model type)
            if hasattr(self.model, "partial_fit"):
                self.model.partial_fit(X_processed, y_new)
                self.logger.info(f"Online update with {len(X_new)} samples")
                return True
            else:
                self.logger.warning("Model does not support online learning")
                return False

        except Exception as e:
            self.logger.error(f"Online update failed: {e}")
            return False

    def detect_data_drift(
        self, X_new: pd.DataFrame, threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect data drift in new features.

        Args:
            X_new: New feature data
            threshold: Drift detection threshold

        Returns:
            Drift detection results
        """
        if self.feature_columns is None:
            return {"drift_detected": False, "message": "No reference data"}

        # Simple drift detection based on feature statistics
        drift_results = {
            "drift_detected": False,
            "drifted_features": [],
            "drift_scores": {},
            "timestamp": datetime.now(),
        }

        try:
            X_processed = self.prepare_features(X_new)

            # Calculate drift scores for each feature
            for feature in self.feature_columns:
                if feature in X_processed.columns:
                    # Use coefficient of variation as simple drift metric
                    new_cv = X_processed[feature].std() / (
                        X_processed[feature].mean() + 1e-8
                    )

                    # This would ideally compare against reference statistics
                    # For now, we use a simple threshold
                    drift_score = abs(new_cv)
                    drift_results["drift_scores"][feature] = drift_score

                    if drift_score > threshold:
                        drift_results["drifted_features"].append(feature)

            drift_results["drift_detected"] = len(drift_results["drifted_features"]) > 0

        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")

        return drift_results

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary.

        Returns:
            Model summary dictionary
        """
        summary = {
            "model_name": self.metadata.model_name if self.metadata else "Unknown",
            "model_type": self.__class__.__name__,
            "version": self.metadata.version if self.metadata else "Unknown",
            "training_date": self.metadata.training_date if self.metadata else None,
            "is_trained": self.model is not None,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "performance_history_length": len(self.performance_history),
            "config": self.config,
        }

        # Add latest performance if available
        if self.performance_history:
            latest_performance = self.performance_history[-1]
            summary["latest_performance"] = latest_performance.to_dict()

        # Add feature importance if available
        importance = self.get_feature_importance()
        if importance:
            # Top 10 most important features
            sorted_importance = sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )
            summary["top_features"] = dict(sorted_importance[:10])

        return summary


class ModelRegistry:
    """
    Model registry for versioning and deployment management.

    Tracks multiple model versions, facilitates A/B testing, and manages
    model lifecycle from development to production deployment.
    """

    def __init__(self, registry_dir: Path = Path("models/registry")):
        """Initialize model registry."""
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.registry_dir / "model_registry.json"
        self.models: Dict[str, List[ModelMetadata]] = {}

        self.logger = logging.getLogger(f"{__name__}.ModelRegistry")

        # Load existing registry
        self._load_registry()

    def _load_registry(self) -> None:
        """Load model registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)

                # Convert to ModelMetadata objects
                for model_name, versions in data.items():
                    self.models[model_name] = [
                        ModelMetadata.from_dict(version_data)
                        for version_data in versions
                    ]

                self.logger.info(f"Loaded registry with {len(self.models)} model types")

            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
                self.models = {}
        else:
            self.models = {}

    def _save_registry(self) -> None:
        """Save model registry to disk."""
        try:
            # Convert to serializable format
            data = {}
            for model_name, versions in self.models.items():
                data[model_name] = [version.to_dict() for version in versions]

            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")

    def register_model(self, metadata: ModelMetadata) -> None:
        """
        Register a new model version.

        Args:
            metadata: Model metadata to register
        """
        model_name = metadata.model_name

        if model_name not in self.models:
            self.models[model_name] = []

        # Check for duplicate versions
        existing_versions = [v.version for v in self.models[model_name]]
        if metadata.version in existing_versions:
            self.logger.warning(
                f"Version {metadata.version} already exists for {model_name}"
            )
            return

        # Add new version
        self.models[model_name].append(metadata)

        # Sort by training date (newest first)
        self.models[model_name].sort(key=lambda x: x.training_date, reverse=True)

        # Save registry
        self._save_registry()

        self.logger.info(f"Registered {model_name} version {metadata.version}")

    def get_model_versions(self, model_name: str) -> List[ModelMetadata]:
        """
        Get all versions of a model.

        Args:
            model_name: Name of the model

        Returns:
            List of model metadata sorted by training date
        """
        return self.models.get(model_name, [])

    def get_latest_model(self, model_name: str) -> Optional[ModelMetadata]:
        """
        Get the latest version of a model.

        Args:
            model_name: Name of the model

        Returns:
            Latest model metadata or None
        """
        versions = self.get_model_versions(model_name)
        return versions[0] if versions else None

    def get_production_model(self, model_name: str) -> Optional[ModelMetadata]:
        """
        Get the current production model.

        Args:
            model_name: Name of the model

        Returns:
            Production model metadata or None
        """
        versions = self.get_model_versions(model_name)
        production_models = [v for v in versions if v.deployment_status == "production"]
        return production_models[0] if production_models else None

    def promote_to_production(self, model_name: str, version: str) -> bool:
        """
        Promote a model version to production.

        Args:
            model_name: Name of the model
            version: Version to promote

        Returns:
            True if promotion was successful
        """
        versions = self.get_model_versions(model_name)

        # Find the target version
        target_version = None
        for v in versions:
            if v.version == version:
                target_version = v
                break

        if target_version is None:
            self.logger.error(f"Version {version} not found for {model_name}")
            return False

        # Demote current production model
        for v in versions:
            if v.deployment_status == "production":
                v.deployment_status = "staging"

        # Promote target version
        target_version.deployment_status = "production"

        # Save changes
        self._save_registry()

        self.logger.info(f"Promoted {model_name} version {version} to production")
        return True

    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get summary of the model registry.

        Returns:
            Registry summary
        """
        summary = {
            "total_model_types": len(self.models),
            "total_versions": sum(len(versions) for versions in self.models.values()),
            "models": {},
        }

        for model_name, versions in self.models.items():
            production_version = None
            latest_version = None

            if versions:
                latest_version = versions[0].version
                production_models = [
                    v for v in versions if v.deployment_status == "production"
                ]
                if production_models:
                    production_version = production_models[0].version

            summary["models"][model_name] = {
                "version_count": len(versions),
                "latest_version": latest_version,
                "production_version": production_version,
            }

        return summary

    def cleanup_old_versions(self, model_name: str, keep_count: int = 5) -> int:
        """
        Clean up old model versions, keeping only the most recent.

        Args:
            model_name: Name of the model
            keep_count: Number of versions to keep

        Returns:
            Number of versions removed
        """
        versions = self.get_model_versions(model_name)

        if len(versions) <= keep_count:
            return 0

        # Keep production model and most recent versions
        production_versions = [
            v for v in versions if v.deployment_status == "production"
        ]
        other_versions = [v for v in versions if v.deployment_status != "production"]

        # Sort other versions by date and keep only the most recent
        other_versions.sort(key=lambda x: x.training_date, reverse=True)
        versions_to_keep = production_versions + other_versions[:keep_count]

        # Update registry
        self.models[model_name] = versions_to_keep
        removed_count = len(versions) - len(versions_to_keep)

        # Save changes
        self._save_registry()

        self.logger.info(f"Cleaned up {removed_count} old versions of {model_name}")
        return removed_count
