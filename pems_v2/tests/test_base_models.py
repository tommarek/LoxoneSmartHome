"""
Test suite for base model infrastructure.

Tests model metadata, registry, and base predictor functionality.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from models.base import (BasePredictor, ModelMetadata, ModelRegistry,
                         PerformanceMetrics, PredictionResult)


class MockPredictor(BasePredictor):
    """Mock predictor for testing base functionality."""

    def train(self, X, y, validation_data=None, **kwargs):
        # Simple mock training
        self.model = MagicMock()
        self.model.predict.return_value = y.values + np.random.normal(0, 0.1, len(y))

        # Fit preprocessing
        self.fit_preprocessing(X)

        # Create mock metadata
        data_hash = self.calculate_data_hash(X, y)
        version = self.generate_version(data_hash)

        self.metadata = ModelMetadata(
            model_name="MockPredictor",
            version=version,
            training_date=datetime.now(),
            features=list(X.columns),
            target_variable="target",
            performance_metrics={"rmse": 0.1, "r2": 0.9},
            training_params=kwargs,
            data_hash=data_hash,
            model_type="Mock",
        )

        return PerformanceMetrics.calculate(
            y.values, self.model.predict(self.prepare_features(X))
        )

    def predict(self, X, return_uncertainty=False, **kwargs):
        if self.model is None:
            raise ValueError("Model not trained")

        X_processed = self.prepare_features(X)
        predictions = pd.Series(
            self.model.predict(X_processed), index=X.index, name="predictions"
        )

        result = PredictionResult(
            predictions=predictions,
            model_version=self.metadata.version if self.metadata else "unknown",
        )

        if return_uncertainty:
            result.uncertainty = pd.Series(
                np.abs(predictions) * 0.1,  # 10% uncertainty
                index=X.index,
                name="uncertainty",
            )

        return result


@pytest.fixture
def sample_data():
    """Create sample training data."""
    dates = pd.date_range("2024-01-01", "2024-01-07", freq="1h")

    X = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, len(dates)),
            "feature2": np.random.normal(5, 2, len(dates)),
            "feature3": np.random.uniform(0, 10, len(dates)),
        },
        index=dates,
    )

    # Create target with some relationship to features
    y = pd.Series(
        X["feature1"] * 2 + X["feature2"] * 0.5 + np.random.normal(0, 0.5, len(dates)),
        index=dates,
        name="target",
    )

    return X, y


@pytest.fixture
def sample_config():
    """Sample configuration for predictors."""
    return {
        "model_dir": "test_models",
        "random_state": 42,
        "validation_split": 0.2,
        "scale_features": True,
        "enable_online_learning": True,
    }


class TestModelMetadata:
    """Test ModelMetadata functionality."""

    def test_metadata_creation(self):
        """Test basic metadata creation."""
        metadata = ModelMetadata(
            model_name="TestModel",
            version="v1.0.0",
            training_date=datetime(2024, 1, 1),
            features=["feature1", "feature2"],
            target_variable="target",
            performance_metrics={"rmse": 0.1, "r2": 0.9},
            training_params={"max_depth": 6},
            data_hash="abc123",
            model_type="XGBoost",
        )

        assert metadata.model_name == "TestModel"
        assert metadata.version == "v1.0.0"
        assert len(metadata.features) == 2
        assert metadata.performance_metrics["rmse"] == 0.1

    def test_metadata_serialization(self):
        """Test metadata to/from dict conversion."""
        original = ModelMetadata(
            model_name="TestModel",
            version="v1.0.0",
            training_date=datetime(2024, 1, 1, 12, 0, 0),
            features=["feature1", "feature2"],
            target_variable="target",
            performance_metrics={"rmse": 0.1},
            training_params={"max_depth": 6},
            data_hash="abc123",
            model_type="XGBoost",
        )

        # Convert to dict
        metadata_dict = original.to_dict()
        assert isinstance(metadata_dict["training_date"], str)
        assert metadata_dict["model_name"] == "TestModel"

        # Convert back
        restored = ModelMetadata.from_dict(metadata_dict)
        assert restored.model_name == original.model_name
        assert restored.training_date == original.training_date
        assert restored.features == original.features


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality."""

    def test_metrics_calculation(self):
        """Test performance metrics calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        metrics = PerformanceMetrics.calculate(y_true, y_pred)

        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert 0 < metrics.r2 <= 1
        assert metrics.mape > 0
        assert abs(metrics.bias) < 1  # Small bias for good predictions

    def test_metrics_edge_cases(self):
        """Test metrics with edge cases."""
        # Empty arrays
        empty_metrics = PerformanceMetrics.calculate(np.array([]), np.array([]))
        assert empty_metrics.mae == float("inf")

        # Perfect predictions
        y_true = np.array([1, 2, 3, 4, 5])
        perfect_metrics = PerformanceMetrics.calculate(y_true, y_true)
        assert perfect_metrics.mae == 0
        assert perfect_metrics.rmse == 0
        assert perfect_metrics.r2 == 1

    def test_metrics_serialization(self):
        """Test metrics to dict conversion."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        metrics = PerformanceMetrics.calculate(y_true, y_pred)
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "mae" in metrics_dict
        assert "rmse" in metrics_dict
        assert "r2" in metrics_dict


class TestBasePredictor:
    """Test BasePredictor functionality."""

    def test_predictor_initialization(self, sample_config):
        """Test predictor initialization."""
        predictor = MockPredictor(sample_config)

        assert predictor.config == sample_config
        assert predictor.random_state == 42
        assert predictor.validation_split == 0.2
        assert predictor.model is None
        assert predictor.metadata is None

    def test_feature_preprocessing(self, sample_config, sample_data):
        """Test feature preprocessing."""
        X, y = sample_data
        predictor = MockPredictor(sample_config)

        # Fit preprocessing
        predictor.fit_preprocessing(X)

        assert predictor.feature_columns == list(X.columns)
        assert predictor.scaler is not None

        # Test feature preparation
        X_processed = predictor.prepare_features(X)
        assert X_processed.shape == X.shape
        assert list(X_processed.columns) == list(X.columns)

        # Check scaling (means should be close to 0)
        assert abs(X_processed.mean()).max() < 0.1

    def test_training_workflow(self, sample_config, sample_data):
        """Test complete training workflow."""
        X, y = sample_data
        predictor = MockPredictor(sample_config)

        # Train model
        performance = predictor.train(X, y)

        assert predictor.model is not None
        assert predictor.metadata is not None
        assert isinstance(performance, PerformanceMetrics)
        assert predictor.metadata.model_name == "MockPredictor"
        assert len(predictor.metadata.features) == len(X.columns)

    def test_prediction_workflow(self, sample_config, sample_data):
        """Test prediction workflow."""
        X, y = sample_data
        predictor = MockPredictor(sample_config)

        # Train first
        predictor.train(X, y)

        # Test prediction
        result = predictor.predict(X[:10])

        assert isinstance(result, PredictionResult)
        assert len(result.predictions) == 10
        assert result.model_version == predictor.metadata.version

        # Test with uncertainty
        result_uncertain = predictor.predict(X[:10], return_uncertainty=True)
        assert result_uncertain.uncertainty is not None
        assert len(result_uncertain.uncertainty) == 10

    def test_model_evaluation(self, sample_config, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        predictor = MockPredictor(sample_config)

        # Train and evaluate
        predictor.train(X, y)
        performance = predictor.evaluate(X, y)

        assert isinstance(performance, PerformanceMetrics)
        assert len(predictor.performance_history) == 1

    def test_data_hash_calculation(self, sample_config, sample_data):
        """Test data hash calculation."""
        X, y = sample_data
        predictor = MockPredictor(sample_config)

        hash1 = predictor.calculate_data_hash(X, y)
        hash2 = predictor.calculate_data_hash(X, y)

        # Same data should produce same hash
        assert hash1 == hash2

        # Different data should produce different hash
        y_modified = y + 0.1
        hash3 = predictor.calculate_data_hash(X, y_modified)
        assert hash1 != hash3

    def test_version_generation(self, sample_config):
        """Test model version generation."""
        predictor = MockPredictor(sample_config)

        version1 = predictor.generate_version("hash123")
        version2 = predictor.generate_version("hash456")

        assert "MockPredictor" in version1
        assert "MockPredictor" in version2
        assert (
            version1 != version2
        )  # Different hashes should produce different versions

    def test_feature_importance(self, sample_config, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        predictor = MockPredictor(sample_config)

        # Mock feature importances
        predictor.train(X, y)
        predictor.model.feature_importances_ = np.array([0.5, 0.3, 0.2])

        importance = predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == len(X.columns)
        assert all(0 <= v <= 1 for v in importance.values())

    def test_model_save_load(self, sample_config, sample_data):
        """Test model saving and loading."""
        X, y = sample_data
        predictor = MockPredictor(sample_config)

        # Train model
        predictor.train(X, y)
        original_version = predictor.metadata.version

        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model.pkl"
            saved_path = predictor.save_model(save_path)

            assert saved_path.exists()
            assert saved_path.with_suffix(".json").exists()  # Metadata file

            # Create new predictor and load
            new_predictor = MockPredictor(sample_config)
            new_predictor.load_model(saved_path)

            assert new_predictor.metadata.version == original_version
            assert new_predictor.feature_columns == predictor.feature_columns
            assert new_predictor.model is not None

    def test_online_learning(self, sample_config, sample_data):
        """Test online learning functionality."""
        X, y = sample_data
        predictor = MockPredictor(sample_config)

        # Train initial model
        predictor.train(X, y)

        # Mock partial_fit capability
        predictor.model.partial_fit = MagicMock()

        # Test online update
        X_new = X[:5]
        y_new = y[:5]

        success = predictor.update_online(X_new, y_new)
        assert success
        assert predictor.model.partial_fit.called

    def test_data_drift_detection(self, sample_config, sample_data):
        """Test data drift detection."""
        X, y = sample_data
        predictor = MockPredictor(sample_config)

        # Train model
        predictor.train(X, y)

        # Test with same data (no drift)
        drift_results = predictor.detect_data_drift(X[:10])
        assert isinstance(drift_results, dict)
        assert "drift_detected" in drift_results
        assert "drifted_features" in drift_results

    def test_model_summary(self, sample_config, sample_data):
        """Test model summary generation."""
        X, y = sample_data
        predictor = MockPredictor(sample_config)

        # Test summary before training
        summary_before = predictor.get_model_summary()
        assert not summary_before["is_trained"]

        # Train and test summary after
        predictor.train(X, y)
        summary_after = predictor.get_model_summary()

        assert summary_after["is_trained"]
        assert summary_after["model_name"] == "MockPredictor"
        assert summary_after["feature_count"] == len(X.columns)
        assert "latest_performance" in summary_after


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir))

            assert registry.registry_dir.exists()
            assert isinstance(registry.models, dict)

    def test_model_registration(self):
        """Test model registration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir))

            metadata = ModelMetadata(
                model_name="TestModel",
                version="v1.0.0",
                training_date=datetime(2024, 1, 1),
                features=["feature1"],
                target_variable="target",
                performance_metrics={"rmse": 0.1},
                training_params={},
                data_hash="hash123",
                model_type="Test",
            )

            registry.register_model(metadata)

            assert "TestModel" in registry.models
            assert len(registry.models["TestModel"]) == 1
            assert registry.models["TestModel"][0].version == "v1.0.0"

    def test_model_versioning(self):
        """Test multiple model versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir))

            # Register multiple versions
            for i in range(3):
                metadata = ModelMetadata(
                    model_name="TestModel",
                    version=f"v1.{i}.0",
                    training_date=datetime(2024, 1, i + 1),
                    features=["feature1"],
                    target_variable="target",
                    performance_metrics={"rmse": 0.1},
                    training_params={},
                    data_hash=f"hash{i}",
                    model_type="Test",
                )
                registry.register_model(metadata)

            versions = registry.get_model_versions("TestModel")
            assert len(versions) == 3

            # Should be sorted by training date (newest first)
            assert versions[0].training_date > versions[1].training_date

    def test_latest_model_retrieval(self):
        """Test getting latest model version."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir))

            # Register models with different dates
            metadata1 = ModelMetadata(
                model_name="TestModel",
                version="v1.0.0",
                training_date=datetime(2024, 1, 1),
                features=[],
                target_variable="target",
                performance_metrics={},
                training_params={},
                data_hash="hash1",
                model_type="Test",
            )

            metadata2 = ModelMetadata(
                model_name="TestModel",
                version="v1.1.0",
                training_date=datetime(2024, 1, 2),
                features=[],
                target_variable="target",
                performance_metrics={},
                training_params={},
                data_hash="hash2",
                model_type="Test",
            )

            registry.register_model(metadata1)
            registry.register_model(metadata2)

            latest = registry.get_latest_model("TestModel")
            assert latest.version == "v1.1.0"

    def test_production_promotion(self):
        """Test promoting models to production."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir))

            # Register model
            metadata = ModelMetadata(
                model_name="TestModel",
                version="v1.0.0",
                training_date=datetime(2024, 1, 1),
                features=[],
                target_variable="target",
                performance_metrics={},
                training_params={},
                data_hash="hash1",
                model_type="Test",
                deployment_status="development",
            )

            registry.register_model(metadata)

            # Promote to production
            success = registry.promote_to_production("TestModel", "v1.0.0")
            assert success

            production_model = registry.get_production_model("TestModel")
            assert production_model.version == "v1.0.0"
            assert production_model.deployment_status == "production"

    def test_registry_persistence(self):
        """Test registry saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create registry and add model
            registry1 = ModelRegistry(Path(temp_dir))

            metadata = ModelMetadata(
                model_name="TestModel",
                version="v1.0.0",
                training_date=datetime(2024, 1, 1),
                features=["feature1"],
                target_variable="target",
                performance_metrics={"rmse": 0.1},
                training_params={},
                data_hash="hash1",
                model_type="Test",
            )

            registry1.register_model(metadata)

            # Create new registry instance (should load saved data)
            registry2 = ModelRegistry(Path(temp_dir))

            assert "TestModel" in registry2.models
            assert len(registry2.models["TestModel"]) == 1
            assert registry2.models["TestModel"][0].version == "v1.0.0"

    def test_registry_summary(self):
        """Test registry summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir))

            # Add multiple models
            for i in range(3):
                metadata = ModelMetadata(
                    model_name=f"Model{i}",
                    version="v1.0.0",
                    training_date=datetime(2024, 1, 1),
                    features=[],
                    target_variable="target",
                    performance_metrics={},
                    training_params={},
                    data_hash=f"hash{i}",
                    model_type="Test",
                )
                registry.register_model(metadata)

            summary = registry.get_registry_summary()

            assert summary["total_model_types"] == 3
            assert summary["total_versions"] == 3
            assert len(summary["models"]) == 3

    def test_version_cleanup(self):
        """Test cleaning up old model versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir))

            # Register many versions
            for i in range(10):
                metadata = ModelMetadata(
                    model_name="TestModel",
                    version=f"v1.{i}.0",
                    training_date=datetime(2024, 1, i + 1),
                    features=[],
                    target_variable="target",
                    performance_metrics={},
                    training_params={},
                    data_hash=f"hash{i}",
                    model_type="Test",
                )
                registry.register_model(metadata)

            # Cleanup, keeping only 3 versions
            removed_count = registry.cleanup_old_versions("TestModel", keep_count=3)

            assert removed_count == 7
            assert len(registry.get_model_versions("TestModel")) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
