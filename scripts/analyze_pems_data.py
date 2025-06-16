#!/usr/bin/env python3
"""
PEMS v2 Data Analysis Script

This script analyzes the behavioral data collected by the PEMS dry run service,
providing insights into optimization performance, control decisions, and system
behavior patterns.

Analysis Features:
1. Optimization performance metrics and trends
2. Control decision patterns and frequency
3. System state evolution and correlations
4. Error analysis and failure patterns
5. Resource usage and efficiency metrics
6. Behavioral anomaly detection
7. Data quality assessment

Usage:
    python analyze_pems_data.py --data-dir ./data/pems_dry_run --output-dir ./analysis
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    optimization_metrics: Dict
    control_patterns: Dict
    system_behavior: Dict
    error_analysis: Dict
    recommendations: List[str]
    data_quality: Dict


class PEMSDataAnalyzer:
    """Main class for analyzing PEMS dry run data."""

    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("PEMSAnalyzer")

        # Data containers
        self.cycle_data: pd.DataFrame = None
        self.mqtt_commands: pd.DataFrame = None
        self.system_states: pd.DataFrame = None
        self.service_metrics: Dict = None
        self.error_log: pd.DataFrame = None

        self.logger.info(f"Initialized analyzer for data in {data_dir}")

    def load_data(self) -> bool:
        """Load all data files from the data directory."""
        try:
            self.logger.info("Loading PEMS behavioral data...")

            # Load optimization cycle data
            cycle_file = self.data_dir / "optimization_cycles.jsonl"
            if cycle_file.exists():
                cycles = []
                with open(cycle_file, "r") as f:
                    for line in f:
                        cycles.append(json.loads(line.strip()))
                self.cycle_data = pd.DataFrame(cycles)
                self.cycle_data["timestamp"] = pd.to_datetime(
                    self.cycle_data["timestamp"]
                )
                self.logger.info(
                    f"  ✅ Loaded {len(self.cycle_data)} optimization cycles"
                )
            else:
                self.logger.warning("  ⚠️ No optimization cycle data found")
                return False

            # Load MQTT commands
            mqtt_file = self.data_dir / "mqtt_commands.jsonl"
            if mqtt_file.exists():
                commands = []
                with open(mqtt_file, "r") as f:
                    for line in f:
                        commands.append(json.loads(line.strip()))
                self.mqtt_commands = pd.DataFrame(commands)
                self.mqtt_commands["timestamp"] = pd.to_datetime(
                    self.mqtt_commands["timestamp"]
                )
                self.logger.info(f"  ✅ Loaded {len(self.mqtt_commands)} MQTT commands")

            # Load system states
            states_file = self.data_dir / "system_states.jsonl"
            if states_file.exists():
                states = []
                with open(states_file, "r") as f:
                    for line in f:
                        states.append(json.loads(line.strip()))
                self.system_states = pd.DataFrame(states)
                if "timestamp" in self.system_states.columns:
                    self.system_states["timestamp"] = pd.to_datetime(
                        self.system_states["timestamp"]
                    )
                self.logger.info(f"  ✅ Loaded {len(self.system_states)} system states")

            # Load service metrics
            metrics_file = self.data_dir / "service_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    self.service_metrics = json.load(f)
                self.logger.info(f"  ✅ Loaded service metrics")

            # Load error log
            errors_file = self.data_dir / "errors.jsonl"
            if errors_file.exists():
                errors = []
                with open(errors_file, "r") as f:
                    for line in f:
                        errors.append(json.loads(line.strip()))
                self.error_log = pd.DataFrame(errors)
                self.error_log["timestamp"] = pd.to_datetime(
                    self.error_log["timestamp"]
                )
                self.logger.info(f"  ✅ Loaded {len(self.error_log)} error records")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return False

    def analyze_optimization_performance(self) -> Dict:
        """Analyze optimization performance metrics."""
        self.logger.info("📊 Analyzing optimization performance...")

        if self.cycle_data is None or len(self.cycle_data) == 0:
            return {"error": "No cycle data available"}

        # Basic statistics
        total_cycles = len(self.cycle_data)
        successful_cycles = self.cycle_data["success"].sum()
        success_rate = successful_cycles / total_cycles if total_cycles > 0 else 0

        # Solve time analysis
        successful_data = self.cycle_data[self.cycle_data["success"]]
        if len(successful_data) > 0:
            avg_solve_time = successful_data["solve_time_seconds"].mean()
            median_solve_time = successful_data["solve_time_seconds"].median()
            p95_solve_time = successful_data["solve_time_seconds"].quantile(0.95)
            min_solve_time = successful_data["solve_time_seconds"].min()
            max_solve_time = successful_data["solve_time_seconds"].max()
        else:
            avg_solve_time = (
                median_solve_time
            ) = p95_solve_time = min_solve_time = max_solve_time = 0

        # Objective value analysis
        if len(successful_data) > 0 and "objective_value" in successful_data.columns:
            finite_objectives = successful_data[
                np.isfinite(successful_data["objective_value"])
            ]
            if len(finite_objectives) > 0:
                avg_objective = finite_objectives["objective_value"].mean()
                objective_std = finite_objectives["objective_value"].std()
                objective_trend = self._calculate_trend(
                    finite_objectives["objective_value"]
                )
            else:
                avg_objective = objective_std = objective_trend = 0
        else:
            avg_objective = objective_std = objective_trend = 0

        # Performance trends over time
        self.cycle_data["hour"] = self.cycle_data["timestamp"].dt.hour
        hourly_performance = (
            self.cycle_data.groupby("hour")
            .agg(
                {
                    "success": "mean",
                    "solve_time_seconds": "mean",
                    "total_cycle_time_ms": "mean",
                }
            )
            .to_dict()
        )

        # Create performance plots
        self._plot_optimization_performance()

        return {
            "total_cycles": total_cycles,
            "successful_cycles": successful_cycles,
            "success_rate": success_rate,
            "solve_time_stats": {
                "mean": avg_solve_time,
                "median": median_solve_time,
                "p95": p95_solve_time,
                "min": min_solve_time,
                "max": max_solve_time,
            },
            "objective_stats": {
                "mean": avg_objective,
                "std": objective_std,
                "trend": objective_trend,
            },
            "hourly_performance": hourly_performance,
        }

    def analyze_control_patterns(self) -> Dict:
        """Analyze control decision patterns."""
        self.logger.info("🎮 Analyzing control patterns...")

        if self.cycle_data is None:
            return {"error": "No cycle data available"}

        # Analyze heating decisions
        heating_patterns = self._analyze_heating_patterns()

        # Analyze Growatt decisions
        growatt_patterns = self._analyze_growatt_patterns()

        # MQTT command analysis
        mqtt_patterns = self._analyze_mqtt_patterns()

        # Create control pattern plots
        self._plot_control_patterns()

        return {
            "heating_patterns": heating_patterns,
            "growatt_patterns": growatt_patterns,
            "mqtt_patterns": mqtt_patterns,
        }

    def analyze_system_behavior(self) -> Dict:
        """Analyze system state evolution and behavior."""
        self.logger.info("🏠 Analyzing system behavior...")

        if self.system_states is None or len(self.system_states) == 0:
            return {"error": "No system state data available"}

        # Temperature evolution
        temp_analysis = self._analyze_temperature_evolution()

        # Battery behavior
        battery_analysis = self._analyze_battery_behavior()

        # Load patterns
        load_analysis = self._analyze_load_patterns()

        # Create system behavior plots
        self._plot_system_behavior()

        return {
            "temperature_analysis": temp_analysis,
            "battery_analysis": battery_analysis,
            "load_analysis": load_analysis,
        }

    def analyze_errors(self) -> Dict:
        """Analyze error patterns and failure modes."""
        self.logger.info("🔍 Analyzing errors and failures...")

        error_stats = {}

        # Optimization failures
        if self.cycle_data is not None:
            failed_cycles = self.cycle_data[~self.cycle_data["success"]]
            error_stats["optimization_failures"] = {
                "count": len(failed_cycles),
                "rate": len(failed_cycles) / len(self.cycle_data)
                if len(self.cycle_data) > 0
                else 0,
            }

            if len(failed_cycles) > 0:
                error_types = failed_cycles["error_type"].value_counts().to_dict()
                error_stats["optimization_failures"]["error_types"] = error_types

        # Service errors
        if self.error_log is not None and len(self.error_log) > 0:
            error_counts = self.error_log["type"].value_counts().to_dict()

            # Recent errors (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_errors = self.error_log[self.error_log["timestamp"] > recent_cutoff]

            error_stats["service_errors"] = {
                "total_count": len(self.error_log),
                "recent_count": len(recent_errors),
                "error_types": error_counts,
            }

        return error_stats

    def assess_data_quality(self) -> Dict:
        """Assess the quality of collected data."""
        self.logger.info("✅ Assessing data quality...")

        quality = {}

        # Cycle data quality
        if self.cycle_data is not None:
            cycle_quality = {
                "total_records": len(self.cycle_data),
                "missing_values": self.cycle_data.isnull().sum().to_dict(),
                "data_span_hours": (
                    self.cycle_data["timestamp"].max()
                    - self.cycle_data["timestamp"].min()
                ).total_seconds()
                / 3600
                if len(self.cycle_data) > 1
                else 0,
                "average_interval_minutes": self._calculate_average_interval(
                    self.cycle_data["timestamp"]
                ),
            }
            quality["cycle_data"] = cycle_quality

        # System state data quality
        if self.system_states is not None:
            state_quality = {
                "total_records": len(self.system_states),
                "missing_values": self.system_states.isnull().sum().to_dict()
                if len(self.system_states) > 0
                else {},
                "data_completeness": len(self.system_states) / len(self.cycle_data)
                if self.cycle_data is not None and len(self.cycle_data) > 0
                else 0,
            }
            quality["system_states"] = state_quality

        return quality

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if self.cycle_data is not None and len(self.cycle_data) > 0:
            success_rate = self.cycle_data["success"].mean()

            if success_rate < 0.8:
                recommendations.append(
                    f"⚠️ Low optimization success rate ({success_rate:.1%}). Consider relaxing solver constraints."
                )

            successful_data = self.cycle_data[self.cycle_data["success"]]
            if len(successful_data) > 0:
                avg_solve_time = successful_data["solve_time_seconds"].mean()
                if avg_solve_time > 10:
                    recommendations.append(
                        f"⚠️ High average solve time ({avg_solve_time:.1f}s). Consider reducing horizon or increasing MIP gap."
                    )

                p95_solve_time = successful_data["solve_time_seconds"].quantile(0.95)
                if p95_solve_time > 30:
                    recommendations.append(
                        f"⚠️ High P95 solve time ({p95_solve_time:.1f}s). Some cycles are very slow."
                    )

        if self.error_log is not None and len(self.error_log) > 0:
            recent_errors = len(
                self.error_log[
                    self.error_log["timestamp"] > datetime.now() - timedelta(hours=24)
                ]
            )
            if recent_errors > 10:
                recommendations.append(
                    f"⚠️ High recent error rate ({recent_errors} errors in 24h)."
                )

        if not recommendations:
            recommendations.append("✅ System is operating within normal parameters.")

        return recommendations

    def _analyze_heating_patterns(self) -> Dict:
        """Analyze heating control patterns."""
        heating_data = []

        for _, cycle in self.cycle_data.iterrows():
            if "heating_decisions" in cycle and cycle["heating_decisions"]:
                for room, decision in cycle["heating_decisions"].items():
                    if isinstance(decision, dict):
                        heating_data.append(
                            {
                                "timestamp": cycle["timestamp"],
                                "room": room,
                                "current_state": decision.get("current_state", False),
                                "duty_cycle": decision.get("duty_cycle", 0),
                            }
                        )

        if not heating_data:
            return {"error": "No heating decision data"}

        heating_df = pd.DataFrame(heating_data)

        # Room-wise statistics
        room_stats = (
            heating_df.groupby("room")
            .agg({"current_state": "mean", "duty_cycle": ["mean", "std"]})
            .round(3)
            .to_dict()
        )

        # Time-based patterns
        heating_df["hour"] = heating_df["timestamp"].dt.hour
        hourly_patterns = (
            heating_df.groupby(["hour", "room"])["current_state"]
            .mean()
            .unstack(fill_value=0)
        )

        return {
            "room_statistics": room_stats,
            "hourly_patterns": hourly_patterns.to_dict()
            if not hourly_patterns.empty
            else {},
        }

    def _analyze_growatt_patterns(self) -> Dict:
        """Analyze Growatt control patterns."""
        growatt_data = []

        for _, cycle in self.cycle_data.iterrows():
            if "growatt_decisions" in cycle and cycle["growatt_decisions"]:
                for mode, decision in cycle["growatt_decisions"].items():
                    if isinstance(decision, dict):
                        growatt_data.append(
                            {
                                "timestamp": cycle["timestamp"],
                                "mode": mode,
                                "current_state": decision.get("current_state", False),
                                "duty_cycle": decision.get("duty_cycle", 0),
                            }
                        )

        if not growatt_data:
            return {"error": "No Growatt decision data"}

        growatt_df = pd.DataFrame(growatt_data)

        # Mode statistics
        mode_stats = (
            growatt_df.groupby("mode")
            .agg({"current_state": "mean", "duty_cycle": ["mean", "std"]})
            .round(3)
            .to_dict()
        )

        return {"mode_statistics": mode_stats}

    def _analyze_mqtt_patterns(self) -> Dict:
        """Analyze MQTT command patterns."""
        if self.mqtt_commands is None or len(self.mqtt_commands) == 0:
            return {"error": "No MQTT command data"}

        # Command frequency by topic
        topic_counts = self.mqtt_commands["topic"].value_counts().to_dict()

        # Commands per hour
        self.mqtt_commands["hour"] = self.mqtt_commands["timestamp"].dt.hour
        hourly_commands = (
            self.mqtt_commands["hour"].value_counts().sort_index().to_dict()
        )

        return {
            "topic_frequencies": topic_counts,
            "hourly_distribution": hourly_commands,
        }

    def _analyze_temperature_evolution(self) -> Dict:
        """Analyze temperature evolution patterns."""
        # This would analyze temperature data from system_states
        # Implementation depends on data structure
        return {"status": "Temperature analysis placeholder"}

    def _analyze_battery_behavior(self) -> Dict:
        """Analyze battery state evolution."""
        if "battery_soc" not in self.cycle_data.columns:
            return {"error": "No battery SOC data"}

        battery_stats = {
            "mean_soc": self.cycle_data["battery_soc"].mean(),
            "min_soc": self.cycle_data["battery_soc"].min(),
            "max_soc": self.cycle_data["battery_soc"].max(),
            "soc_std": self.cycle_data["battery_soc"].std(),
        }

        return battery_stats

    def _analyze_load_patterns(self) -> Dict:
        """Analyze load patterns."""
        if "current_load" not in self.cycle_data.columns:
            return {"error": "No load data"}

        load_stats = {
            "mean_load": self.cycle_data["current_load"].mean(),
            "min_load": self.cycle_data["current_load"].min(),
            "max_load": self.cycle_data["current_load"].max(),
            "load_std": self.cycle_data["current_load"].std(),
        }

        return load_stats

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend slope."""
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series, 1)
        return coeffs[0]  # Slope

    def _calculate_average_interval(self, timestamps: pd.Series) -> float:
        """Calculate average interval between timestamps in minutes."""
        if len(timestamps) < 2:
            return 0
        intervals = timestamps.diff().dropna()
        return intervals.mean().total_seconds() / 60

    def _plot_optimization_performance(self):
        """Create optimization performance plots."""
        if self.cycle_data is None or len(self.cycle_data) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("PEMS v2 Optimization Performance Analysis", fontsize=16)

        # Success rate over time
        self.cycle_data.set_index("timestamp")["success"].rolling("1H").mean().plot(
            ax=axes[0, 0],
            title="Success Rate Over Time (1h rolling)",
            ylabel="Success Rate",
        )
        axes[0, 0].set_ylim(0, 1)

        # Solve time distribution
        successful_data = self.cycle_data[self.cycle_data["success"]]
        if len(successful_data) > 0:
            successful_data["solve_time_seconds"].hist(bins=30, ax=axes[0, 1])
            axes[0, 1].set_title("Solve Time Distribution")
            axes[0, 1].set_xlabel("Solve Time (seconds)")
            axes[0, 1].axvline(
                successful_data["solve_time_seconds"].mean(),
                color="red",
                linestyle="--",
                label="Mean",
            )
            axes[0, 1].legend()

        # Objective value over time
        finite_objectives = self.cycle_data[
            np.isfinite(self.cycle_data["objective_value"])
        ]
        if len(finite_objectives) > 0:
            finite_objectives.set_index("timestamp")["objective_value"].plot(
                ax=axes[1, 0], title="Objective Value Over Time"
            )

        # Hourly performance
        hourly_success = self.cycle_data.groupby(self.cycle_data["timestamp"].dt.hour)[
            "success"
        ].mean()
        hourly_success.plot(kind="bar", ax=axes[1, 1], title="Success Rate by Hour")
        axes[1, 1].set_xlabel("Hour of Day")
        axes[1, 1].set_ylabel("Success Rate")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "optimization_performance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_control_patterns(self):
        """Create control pattern plots."""
        # This would create visualizations of control decisions
        pass

    def _plot_system_behavior(self):
        """Create system behavior plots."""
        if self.cycle_data is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("PEMS v2 System Behavior Analysis", fontsize=16)

        # Battery SOC evolution
        if "battery_soc" in self.cycle_data.columns:
            self.cycle_data.set_index("timestamp")["battery_soc"].plot(
                ax=axes[0, 0], title="Battery SOC Evolution"
            )
            axes[0, 0].set_ylabel("SOC")

        # Load evolution
        if "current_load" in self.cycle_data.columns:
            self.cycle_data.set_index("timestamp")["current_load"].plot(
                ax=axes[0, 1], title="Load Evolution"
            )
            axes[0, 1].set_ylabel("Load (W)")

        # Temperature evolution (outdoor)
        if "outdoor_temp" in self.cycle_data.columns:
            self.cycle_data.set_index("timestamp")["outdoor_temp"].plot(
                ax=axes[1, 0], title="Outdoor Temperature"
            )
            axes[1, 0].set_ylabel("Temperature (°C)")

        # PV power
        if "pv_power" in self.cycle_data.columns:
            self.cycle_data.set_index("timestamp")["pv_power"].plot(
                ax=axes[1, 1], title="PV Power"
            )
            axes[1, 1].set_ylabel("PV Power (W)")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "system_behavior.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def run_complete_analysis(self) -> AnalysisResult:
        """Run complete analysis and generate report."""
        self.logger.info("🚀 Starting complete PEMS data analysis")

        if not self.load_data():
            raise ValueError("Failed to load data")

        # Run all analyses
        optimization_metrics = self.analyze_optimization_performance()
        control_patterns = self.analyze_control_patterns()
        system_behavior = self.analyze_system_behavior()
        error_analysis = self.analyze_errors()
        data_quality = self.assess_data_quality()
        recommendations = self.generate_recommendations()

        # Create analysis result
        result = AnalysisResult(
            optimization_metrics=optimization_metrics,
            control_patterns=control_patterns,
            system_behavior=system_behavior,
            error_analysis=error_analysis,
            recommendations=recommendations,
            data_quality=data_quality,
        )

        # Generate report
        self._generate_report(result)

        self.logger.info("✅ Analysis complete")
        return result

    def _generate_report(self, result: AnalysisResult):
        """Generate comprehensive analysis report."""
        report_file = self.output_dir / "pems_analysis_report.md"

        with open(report_file, "w") as f:
            f.write("# PEMS v2 Behavioral Data Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            if "success_rate" in result.optimization_metrics:
                success_rate = result.optimization_metrics["success_rate"]
                f.write(f"- **Optimization Success Rate**: {success_rate:.1%}\n")

            if "solve_time_stats" in result.optimization_metrics:
                avg_solve = result.optimization_metrics["solve_time_stats"]["mean"]
                f.write(f"- **Average Solve Time**: {avg_solve:.1f} seconds\n")

            f.write(
                f"- **Total Optimization Cycles**: {result.optimization_metrics.get('total_cycles', 0)}\n"
            )
            f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            for rec in result.recommendations:
                f.write(f"- {rec}\n")
            f.write("\n")

            # Detailed Results
            f.write("## Detailed Analysis Results\n\n")
            f.write("### Optimization Performance\n")
            f.write(
                f"```json\n{json.dumps(result.optimization_metrics, indent=2)}\n```\n\n"
            )

            f.write("### Control Patterns\n")
            f.write(
                f"```json\n{json.dumps(result.control_patterns, indent=2)}\n```\n\n"
            )

            f.write("### Error Analysis\n")
            f.write(f"```json\n{json.dumps(result.error_analysis, indent=2)}\n```\n\n")

            f.write("### Data Quality\n")
            f.write(f"```json\n{json.dumps(result.data_quality, indent=2)}\n```\n\n")

        self.logger.info(f"📄 Report saved to {report_file}")


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze PEMS v2 behavioral data")
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing PEMS data files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for analysis output"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    setup_logging()

    try:
        analyzer = PEMSDataAnalyzer(args.data_dir, args.output_dir)
        result = analyzer.run_complete_analysis()

        print("\n🎉 Analysis completed successfully!")
        print(f"📄 Report generated in: {args.output_dir}")
        print(
            f"📊 Success rate: {result.optimization_metrics.get('success_rate', 0):.1%}"
        )
        print(f"🔍 Recommendations: {len(result.recommendations)}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
