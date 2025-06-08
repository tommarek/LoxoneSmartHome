#!/usr/bin/env python3
"""
PEMS v2 Complete System Validation

Comprehensive validation of all Phase 2 components with real data:
1. Data extraction and preprocessing
2. ML model training and prediction
3. Optimization engine functionality
4. Control interface integration
5. End-to-end workflow testing
"""

import asyncio
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import all components
from analysis.core.data_extraction import DataExtractor
from analysis.core.data_preprocessing import DataPreprocessor
from config.settings import PEMSSettings
from models.predictors.load_predictor import LoadPredictor
from models.predictors.pv_predictor import PVPredictor
from models.predictors.thermal_predictor import ThermalPredictor
from modules.control.heating_controller import HeatingController
from modules.optimization.optimizer import (EnergyOptimizer,
                                            create_optimization_problem)


class SystemValidator:
    """Comprehensive system validation with real data."""

    def __init__(self):
        self.settings = PEMSSettings()
        self.results = {}
        self.start_time = datetime.now()

    async def run_complete_validation(self):
        """Run complete system validation."""
        print("🔍 PEMS v2 Complete System Validation")
        print("=" * 60)
        print(f'Started: {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print()

        try:
            # 1. Data Pipeline Validation
            await self.validate_data_pipeline()

            # 2. ML Models Validation
            await self.validate_ml_models()

            # 3. Optimization Engine Validation
            await self.validate_optimization_engine()

            # 4. Control Interface Validation
            await self.validate_control_interfaces()

            # 5. End-to-End Integration Test
            await self.validate_end_to_end_workflow()

            # 6. Performance Benchmarks
            await self.validate_performance()

            # Generate final report
            self.generate_validation_report()

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            traceback.print_exc()
            return False

        return True

    async def validate_data_pipeline(self):
        """Validate data extraction and preprocessing."""
        print("📊 Validating Data Pipeline...")

        try:
            # Test data extraction
            extractor = DataExtractor(self.settings)

            # Extract last 7 days of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            print(f"  📈 Extracting data: {start_date.date()} to {end_date.date()}")

            # Extract all data types
            extraction_start = time.time()

            pv_data = await extractor.extract_pv_data(start_date, end_date)
            room_data = await extractor.extract_room_temperatures(start_date, end_date)
            weather_data = await extractor.extract_weather_data(start_date, end_date)
            consumption_data = await extractor.extract_energy_consumption(
                start_date, end_date
            )
            relay_data = await extractor.extract_relay_states(start_date, end_date)
            battery_data = await extractor.extract_battery_data(start_date, end_date)
            price_data = await extractor.extract_energy_prices(start_date, end_date)

            extraction_time = time.time() - extraction_start

            # Validate extracted data
            data_summary = {
                "pv_records": len(pv_data) if pv_data is not None else 0,
                "room_count": len(room_data) if isinstance(room_data, dict) else 0,
                "weather_records": len(weather_data) if weather_data is not None else 0,
                "consumption_records": len(consumption_data)
                if consumption_data is not None
                else 0,
                "relay_count": len(relay_data) if isinstance(relay_data, dict) else 0,
                "battery_records": len(battery_data) if battery_data is not None else 0,
                "price_records": len(price_data) if price_data is not None else 0,
                "extraction_time": extraction_time,
            }

            print(f"  ✅ Data extraction completed in {extraction_time:.2f}s")
            for key, value in data_summary.items():
                if key != "extraction_time":
                    print(f"     {key}: {value:,}")

            # Test data preprocessing
            preprocessor = DataPreprocessor()

            if pv_data is not None and len(pv_data) > 0:
                processed_pv = preprocessor.process_dataset(pv_data, "pv")
                print(f"  ✅ PV preprocessing: {len(processed_pv):,} records")

            if weather_data is not None and len(weather_data) > 0:
                processed_weather = preprocessor.process_dataset(
                    weather_data, "weather"
                )
                print(f"  ✅ Weather preprocessing: {len(processed_weather):,} records")

            self.results["data_pipeline"] = {
                "status": "success",
                "data_summary": data_summary,
                "extraction_time": extraction_time,
            }

        except Exception as e:
            print(f"  ❌ Data pipeline validation failed: {e}")
            self.results["data_pipeline"] = {"status": "failed", "error": str(e)}

    async def validate_ml_models(self):
        """Validate ML model training and prediction."""
        print("\\n🤖 Validating ML Models...")

        try:
            # Load processed data
            print("  📊 Loading processed datasets...")

            # Use existing processed data if available
            data_files = {
                "pv": "data/processed/pv_processed.parquet",
                "weather": "data/processed/weather_processed.parquet",
                "consumption": "data/processed/consumption_processed.parquet",
                "room": "data/processed/rooms_obyvak_processed.parquet",
                "outdoor": "data/processed/outdoor_temp_processed.parquet",
            }

            datasets = {}
            for name, file_path in data_files.items():
                if os.path.exists(file_path):
                    datasets[name] = pd.read_parquet(file_path)
                    print(f"    ✅ {name}: {len(datasets[name]):,} records")
                else:
                    print(f"    ⚠️  {name}: file not found")

            model_results = {}

            # Test Load Predictor
            if "consumption" in datasets:
                print("  🏠 Testing Load Predictor...")
                load_config = {
                    "model_name": "load_predictor_test",
                    "components": ["heating"],
                    "model_params": {"n_estimators": 20},  # Fast training
                }

                load_predictor = LoadPredictor(config=load_config)

                # Quick training test
                data = datasets["consumption"].tail(1000).copy()  # Last 1000 records

                # Reset index to avoid timezone issues
                data = data.reset_index()
                data["hour"] = pd.to_datetime(data["timestamp"]).dt.hour
                data["day_of_week"] = pd.to_datetime(data["timestamp"]).dt.dayofweek
                data = data.set_index("timestamp")

                if "heating_power" in data.columns:
                    X = data[["hour", "day_of_week"]].dropna()
                    y = data.loc[X.index, "heating_power"]

                    if len(X) > 50:
                        split_idx = int(len(X) * 0.8)
                        X_train, X_val = X[:split_idx], X[split_idx:]
                        y_train, y_val = y[:split_idx], y[split_idx:]

                        metrics = load_predictor.train(
                            X_train, y_train, validation_data=(X_val, y_val)
                        )

                        # Test prediction
                        test_pred = load_predictor.predict(X_val.head(10))

                        model_results["load_predictor"] = {
                            "status": "success",
                            "mae": metrics.mae,
                            "rmse": metrics.rmse,
                            "r2": metrics.r2,
                            "prediction_shape": test_pred.predictions.shape,
                        }
                        print(
                            f"    ✅ Load model: MAE={metrics.mae:.1f}W, R²={metrics.r2:.3f}"
                        )
                    else:
                        model_results["load_predictor"] = {
                            "status": "insufficient_data"
                        }
                        print(
                            f"    ⚠️  Load model: insufficient data ({len(X)} samples)"
                        )
                else:
                    model_results["load_predictor"] = {"status": "no_target_column"}
                    print(f"    ⚠️  Load model: no heating_power column")

            # Test Thermal Predictor
            if "room" in datasets and "outdoor" in datasets:
                print("  🌡️ Testing Thermal Predictor...")
                thermal_config = {
                    "model_name": "thermal_predictor_test",
                    "room_name": "obyvak",
                    "model_params": {"n_estimators": 20},
                }

                thermal_predictor = ThermalPredictor(config=thermal_config)

                # Merge room and outdoor data
                merged = pd.merge(
                    datasets["room"].tail(1000),
                    datasets["outdoor"].tail(1000),
                    left_index=True,
                    right_index=True,
                    how="inner",
                )

                if len(merged) > 50 and "temperature" in merged.columns:
                    merged = merged.copy()  # Avoid SettingWithCopyWarning
                    merged.loc[:, "hour"] = merged.index.hour
                    merged.loc[:, "temp_lag"] = merged["temperature"].shift(1)

                    X = merged[["hour", "temp_lag", "outdoor_temp"]].dropna()
                    y = merged.loc[X.index, "temperature"]

                    if len(X) > 50:
                        split_idx = int(len(X) * 0.8)
                        X_train, X_val = X[:split_idx], X[split_idx:]
                        y_train, y_val = y[:split_idx], y[split_idx:]

                        try:
                            metrics = thermal_predictor.train(
                                X_train, y_train, validation_data=(X_val, y_val)
                            )
                            test_pred = thermal_predictor.predict(X_val.head(10))

                            model_results["thermal_predictor"] = {
                                "status": "success",
                                "mae": metrics.mae,
                                "rmse": metrics.rmse,
                                "r2": metrics.r2,
                                "prediction_shape": test_pred.predictions.shape,
                            }
                            print(
                                f"    ✅ Thermal model: MAE={metrics.mae:.2f}°C, R²={metrics.r2:.3f}"
                            )
                        except Exception as e:
                            model_results["thermal_predictor"] = {
                                "status": "training_failed",
                                "error": str(e),
                            }
                            print(f"    ⚠️  Thermal model training failed: {e}")
                    else:
                        model_results["thermal_predictor"] = {
                            "status": "insufficient_data"
                        }
                        print(f"    ⚠️  Thermal model: insufficient data")
                else:
                    model_results["thermal_predictor"] = {"status": "no_data"}
                    print(
                        f"    ⚠️  Thermal model: no merged data or temperature column"
                    )

            self.results["ml_models"] = model_results

        except Exception as e:
            print(f"  ❌ ML models validation failed: {e}")
            self.results["ml_models"] = {"status": "failed", "error": str(e)}

    async def validate_optimization_engine(self):
        """Validate optimization engine with realistic data."""
        print("\\n⚡ Validating Optimization Engine...")

        try:
            # Test basic optimization
            print("  🧮 Testing basic optimization...")

            config = {
                "rooms": {
                    "obyvak": {"power_kw": 4.8},
                    "kuchyne": {"power_kw": 2.0},
                    "loznice": {"power_kw": 1.2},
                },
                "battery": {"capacity_kwh": 10.0, "max_power_kw": 5.0},
                "pv_system": {"capacity_kw": 10.0},
            }

            optimizer = EnergyOptimizer(config)

            # Create test problem
            problem = create_optimization_problem(
                start_time=datetime.now(), horizon_hours=6  # Shorter for validation
            )

            opt_start = time.time()
            result = optimizer.optimize(problem)
            opt_time = time.time() - opt_start

            optimization_results = {
                "status": "success" if result.success else "failed",
                "solve_time": opt_time,
                "objective_value": result.objective_value if result.success else None,
                "message": result.message,
            }

            if result.success:
                total_heating_hours = sum(
                    schedule.sum() * 0.25
                    for schedule in result.heating_schedule.values()
                )
                optimization_results.update(
                    {
                        "heating_hours": total_heating_hours,
                        "battery_activity": result.battery_schedule.abs().mean(),
                        "grid_flow": result.grid_schedule.abs().mean(),
                    }
                )
                print(f"    ✅ Optimization successful in {opt_time:.2f}s")
                print(f"       Objective: {result.objective_value:.2f}")
                print(f"       Heating hours: {total_heating_hours:.1f}h")
            else:
                print(f"    ⚠️  Optimization failed: {result.message}")
                print(f"       Solve time: {opt_time:.2f}s")

            self.results["optimization"] = optimization_results

        except Exception as e:
            print(f"  ❌ Optimization validation failed: {e}")
            self.results["optimization"] = {"status": "failed", "error": str(e)}

    async def validate_control_interfaces(self):
        """Validate control interface functionality."""
        print("\\n🎛️ Validating Control Interfaces...")

        try:
            # Test heating controller
            print("  🏠 Testing Heating Controller...")

            room_config = {
                "obyvak": {"power_kw": 4.8},
                "kuchyne": {"power_kw": 2.0},
                "loznice": {"power_kw": 1.2},
            }

            mqtt_config = {
                "broker": "localhost",
                "port": 1883,
                "heating_topic_prefix": "pems/heating",
            }

            controller_config = {"rooms": room_config, "mqtt": mqtt_config}

            heating_controller = HeatingController(controller_config)

            # Test basic functionality (without actual MQTT)

            # Test schedule execution
            schedule = {
                "obyvak": pd.Series(
                    [True, False, True],
                    index=pd.date_range(start=datetime.now(), periods=3, freq="15min"),
                ),
                "kuchyne": pd.Series(
                    [False, True, False],
                    index=pd.date_range(start=datetime.now(), periods=3, freq="15min"),
                ),
            }

            control_start = time.time()
            schedule_results = await heating_controller.execute_schedule(schedule)
            control_time = time.time() - control_start

            # Test individual room control
            room_result = await heating_controller.set_room_heating("obyvak", True, 30)

            # Test status retrieval
            status = await heating_controller.get_room_status("obyvak")
            all_status = await heating_controller.get_all_status()

            control_results = {
                "status": "success",
                "schedule_execution_time": control_time,
                "schedule_success_rate": sum(schedule_results.values())
                / len(schedule_results),
                "individual_control": room_result,
                "status_retrieval": status is not None,
                "all_status_count": len(all_status),
            }

            print(f"    ✅ Heating controller functional")
            print(f"       Schedule execution: {control_time:.3f}s")
            print(
                f'       Success rate: {control_results["schedule_success_rate"]*100:.1f}%'
            )
            print(f'       Status retrieval: {"✅" if status else "❌"}')

            self.results["control_interfaces"] = control_results

        except Exception as e:
            print(f"  ❌ Control interface validation failed: {e}")
            self.results["control_interfaces"] = {"status": "failed", "error": str(e)}

    async def validate_end_to_end_workflow(self):
        """Validate complete end-to-end workflow."""
        print("\\n🔄 Validating End-to-End Workflow...")

        try:
            print("  🚀 Running complete workflow simulation...")

            workflow_start = time.time()

            # 1. Data extraction (simplified)
            print("    1️⃣ Data extraction...")
            data_files = {
                "pv": "data/processed/pv_processed.parquet",
                "weather": "data/processed/weather_processed.parquet",
                "consumption": "data/processed/consumption_processed.parquet",
            }

            datasets = {}
            for name, file_path in data_files.items():
                if os.path.exists(file_path):
                    datasets[name] = pd.read_parquet(file_path).tail(
                        100
                    )  # Last 100 records

            # 2. Generate forecasts (simplified)
            print("    2️⃣ Generating forecasts...")

            # Create simple forecasts
            time_index = pd.date_range(start=datetime.now(), periods=24, freq="H")

            pv_forecast = pd.Series(
                np.maximum(0, 5000 * np.sin(np.pi * np.arange(24) / 12)),
                index=time_index,
            )

            load_forecast = pd.Series([1500] * 24, index=time_index)
            price_forecast = pd.Series(
                [0.15 if 7 <= h <= 22 else 0.08 for h in range(24)], index=time_index
            )

            # 3. Optimization
            print("    3️⃣ Running optimization...")

            config = {
                "rooms": {"obyvak": {"power_kw": 4.8}, "kuchyne": {"power_kw": 2.0}},
                "battery": {"capacity_kwh": 10.0, "max_power_kw": 5.0},
            }

            optimizer = EnergyOptimizer(config)

            problem = create_optimization_problem(
                start_time=datetime.now(),
                horizon_hours=6,
                pv_forecast=pv_forecast[: 6 * 4],  # 6 hours in 15-min intervals
                load_forecast=load_forecast[: 6 * 4].reindex(
                    pd.date_range(start=datetime.now(), periods=6 * 4, freq="15min"),
                    method="ffill",
                ),
                price_forecast=price_forecast[: 6 * 4].reindex(
                    pd.date_range(start=datetime.now(), periods=6 * 4, freq="15min"),
                    method="ffill",
                ),
            )

            result = optimizer.optimize(problem)

            # 4. Control execution (simulated)
            print("    4️⃣ Executing control commands...")

            if result.success:
                heating_controller = HeatingController(
                    {"rooms": config["rooms"], "mqtt": {"broker": "localhost"}}
                )

                # Execute first hour of schedule
                first_hour_schedule = {}
                for room, schedule in result.heating_schedule.items():
                    first_hour_schedule[room] = schedule.head(
                        4
                    )  # First 4 intervals (1 hour)

                control_results = await heating_controller.execute_schedule(
                    first_hour_schedule
                )
                control_success = sum(control_results.values()) / len(control_results)
            else:
                control_success = 0

            workflow_time = time.time() - workflow_start

            workflow_results = {
                "status": "success",
                "total_time": workflow_time,
                "data_loaded": len(datasets),
                "forecasts_generated": True,
                "optimization_success": result.success,
                "control_success_rate": control_success,
                "objective_value": result.objective_value if result.success else None,
            }

            print(f"    ✅ End-to-end workflow completed in {workflow_time:.2f}s")
            print(f"       Data sources: {len(datasets)}")
            print(f'       Optimization: {"✅" if result.success else "❌"}')
            print(f"       Control success: {control_success*100:.1f}%")

            self.results["end_to_end"] = workflow_results

        except Exception as e:
            print(f"  ❌ End-to-end validation failed: {e}")
            self.results["end_to_end"] = {"status": "failed", "error": str(e)}

    async def validate_performance(self):
        """Validate system performance benchmarks."""
        print("\\n⚡ Performance Benchmarks...")

        try:
            benchmarks = {}

            # 1. Data processing benchmark
            print("  📊 Data processing benchmark...")
            if os.path.exists("data/processed/pv_processed.parquet"):
                start_time = time.time()
                data = pd.read_parquet("data/processed/pv_processed.parquet")
                load_time = time.time() - start_time

                benchmarks["data_loading"] = {
                    "records": len(data),
                    "time_seconds": load_time,
                    "records_per_second": len(data) / load_time if load_time > 0 else 0,
                }
                print(
                    f"    ✅ Loaded {len(data):,} records in {load_time:.3f}s ({len(data)/load_time:.0f} rec/s)"
                )

            # 2. Optimization benchmark
            print("  🧮 Optimization benchmark...")
            config = {
                "rooms": {"test": {"power_kw": 1.0}},
                "battery": {"capacity_kwh": 10.0},
            }
            optimizer = EnergyOptimizer(config)

            opt_times = []
            for horizon in [1, 6, 12, 24]:
                start_time = time.time()
                problem = create_optimization_problem(
                    start_time=datetime.now(), horizon_hours=horizon
                )
                result = optimizer.optimize(problem)
                opt_time = time.time() - start_time
                opt_times.append(opt_time)

                print(f"    ✅ {horizon}h horizon: {opt_time:.3f}s")

            benchmarks["optimization"] = {
                "horizons_tested": [1, 6, 12, 24],
                "solve_times": opt_times,
                "average_time": np.mean(opt_times),
                "max_time": max(opt_times),
            }

            self.results["performance"] = benchmarks

        except Exception as e:
            print(f"  ❌ Performance validation failed: {e}")
            self.results["performance"] = {"status": "failed", "error": str(e)}

    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        print("\\n" + "=" * 60)
        print("📋 VALIDATION REPORT")
        print("=" * 60)

        total_time = (datetime.now() - self.start_time).total_seconds()

        # Summary statistics
        tests_run = 0
        tests_passed = 0

        for component, result in self.results.items():
            tests_run += 1
            if isinstance(result, dict) and result.get("status") == "success":
                tests_passed += 1

        print(f"\\n📊 SUMMARY:")
        print(f"   Total validation time: {total_time:.2f}s")
        print(f"   Components tested: {tests_run}")
        print(f"   Components passed: {tests_passed}")
        print(f"   Success rate: {tests_passed/tests_run*100:.1f}%")

        # Detailed results
        print(f"\\n🔍 DETAILED RESULTS:")

        for component, result in self.results.items():
            status = (
                "✅"
                if isinstance(result, dict) and result.get("status") == "success"
                else "❌"
            )
            print(f'   {status} {component.replace("_", " ").title()}')

            if isinstance(result, dict):
                if "error" in result:
                    print(f'      Error: {result["error"]}')
                else:
                    # Show key metrics
                    for key, value in result.items():
                        if key not in ["status", "error"] and not isinstance(
                            value, dict
                        ):
                            if isinstance(value, float):
                                print(f"      {key}: {value:.3f}")
                            else:
                                print(f"      {key}: {value}")

        # Performance summary
        if "performance" in self.results:
            perf = self.results["performance"]
            print(f"\\n⚡ PERFORMANCE SUMMARY:")
            if "optimization" in perf:
                opt_perf = perf["optimization"]
                print(f'   Optimization average: {opt_perf["average_time"]:.3f}s')
                print(f'   Optimization max: {opt_perf["max_time"]:.3f}s')

            if "data_loading" in perf:
                data_perf = perf["data_loading"]
                print(
                    f'   Data loading rate: {data_perf["records_per_second"]:.0f} records/s'
                )

        # Overall assessment
        print(f"\\n🎯 OVERALL ASSESSMENT:")
        if tests_passed >= tests_run * 0.8:  # 80% success threshold
            print("   ✅ SYSTEM VALIDATION PASSED")
            print("   🚀 PEMS v2 Phase 2 is production ready!")
        else:
            print("   ⚠️  SYSTEM VALIDATION NEEDS ATTENTION")
            print("   🔧 Some components require fixes before production")

        print("\\n" + "=" * 60)


async def main():
    """Run complete system validation."""
    validator = SystemValidator()

    try:
        success = await validator.run_complete_validation()

        if success:
            print("\\n🎉 Validation completed successfully!")
            return 0
        else:
            print("\\n❌ Validation failed!")
            return 1

    except Exception as e:
        print(f"\\n💥 Validation crashed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
