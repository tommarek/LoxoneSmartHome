"""
Visualization utilities for PEMS v2 data analysis.
Adapted for PV system with conditional export capability.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


class AnalysisVisualizer:
    """Create visualizations for analysis results."""

    def __init__(self, output_dir: str = "analysis/figures"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

        # Your system's export threshold (update as needed)
        self.export_price_threshold = 100  # CZK/MWh or your unit

    def plot_pv_analysis_dashboard(
        self,
        pv_data: pd.DataFrame,
        analysis_results: Dict[str, Any],
        price_data: Optional[pd.DataFrame] = None,
        export_enabled_date: Optional[datetime] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive PV analysis dashboard adapted for conditional export.

        Special handling for your system:
        - Pre-export period: Focus on self-consumption metrics
        - Post-export period: Show price-based export decisions
        """
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "PV Production & Self-Consumption",
                "Export Decision vs Price (Post-Export Period)",
                "Battery Cycling Patterns",
                "Daily Energy Balance",
                "Self-Consumption Ratio Evolution",
                "Economic Performance",
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": True}],
            ],
        )

        # 1. PV Production & Self-Consumption
        if export_enabled_date:
            pre_export = pv_data[pv_data.index < export_enabled_date]
            post_export = pv_data[pv_data.index >= export_enabled_date]

            # Pre-export period (gray background)
            fig.add_vrect(
                x0=pre_export.index.min(),
                x1=pre_export.index.max(),
                fillcolor="gray",
                opacity=0.1,
                annotation_text="Export Disabled",
                row=1,
                col=1,
            )

        # Add PV production
        fig.add_trace(
            go.Scatter(
                x=pv_data.index,
                y=pv_data["InputPower"],
                name="PV Production",
                line=dict(color="gold", width=2),
            ),
            row=1,
            col=1,
        )

        # Add self-consumption
        if "SelfConsumption" in pv_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=pv_data.index,
                    y=pv_data["SelfConsumption"],
                    name="Self-Consumption",
                    line=dict(color="green", width=2),
                ),
                row=1,
                col=1,
            )

        # 2. Export Decision vs Price (Post-Export Period only)
        if price_data is not None and export_enabled_date:
            post_export_price = price_data[price_data.index >= export_enabled_date]
            post_export_pv = pv_data[pv_data.index >= export_enabled_date]

            # Export power vs price scatter
            if "ExportPower" in post_export_pv.columns:
                merged_data = post_export_pv.merge(
                    post_export_price, left_index=True, right_index=True, how="inner"
                )

                fig.add_trace(
                    go.Scatter(
                        x=merged_data["price"],
                        y=merged_data["ExportPower"],
                        mode="markers",
                        name="Export vs Price",
                        marker=dict(
                            color=merged_data["ExportPower"],
                            colorscale="Viridis",
                            size=6,
                        ),
                    ),
                    row=1,
                    col=2,
                )

                # Add price threshold line
                fig.add_hline(
                    y=self.export_price_threshold,
                    line=dict(color="red", dash="dash"),
                    annotation_text="Export Threshold",
                    row=1,
                    col=2,
                )

        # 3. Battery Cycling Patterns
        if "BatterySOC" in pv_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=pv_data.index,
                    y=pv_data["BatterySOC"],
                    name="Battery SOC (%)",
                    line=dict(color="blue", width=2),
                ),
                row=2,
                col=1,
            )

        # 4. Daily Energy Balance
        daily_data = pv_data.resample("D").sum()
        fig.add_trace(
            go.Bar(
                x=daily_data.index,
                y=daily_data["InputPower"],
                name="Daily PV Production",
                marker_color="gold",
            ),
            row=2,
            col=2,
        )

        if "SelfConsumption" in daily_data.columns:
            fig.add_trace(
                go.Bar(
                    x=daily_data.index,
                    y=daily_data["SelfConsumption"],
                    name="Daily Self-Consumption",
                    marker_color="green",
                ),
                row=2,
                col=2,
            )

        # 5. Self-Consumption Ratio Evolution
        if "SelfConsumption" in pv_data.columns:
            daily_ratio = (
                daily_data["SelfConsumption"] / daily_data["InputPower"]
            ).fillna(0)
            fig.add_trace(
                go.Scatter(
                    x=daily_ratio.index,
                    y=daily_ratio * 100,
                    name="Self-Consumption Ratio (%)",
                    line=dict(color="purple", width=2),
                ),
                row=3,
                col=1,
            )

        # 6. Economic Performance
        if "economic_analysis" in analysis_results:
            econ_data = analysis_results["economic_analysis"]

            if "monthly_savings" in econ_data:
                monthly_savings = pd.Series(econ_data["monthly_savings"])
                fig.add_trace(
                    go.Bar(
                        x=monthly_savings.index,
                        y=monthly_savings.values,
                        name="Monthly Savings (CZK)",
                        marker_color="darkgreen",
                    ),
                    row=3,
                    col=2,
                )

        # Update layout
        fig.update_layout(
            title="PV System Analysis Dashboard - Conditional Export Configuration",
            height=1200,
            showlegend=True,
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_thermal_analysis(
        self,
        room_data: Dict[str, pd.DataFrame],
        rc_parameters: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create thermal analysis visualization for relay-controlled rooms.

        Args:
            room_data: Dictionary with room names as keys, temperature data as values
            rc_parameters: RC thermal parameters for each room
            save_path: Optional path to save the figure
        """
        num_rooms = len(room_data)
        cols = min(3, num_rooms)
        rows = (num_rooms + cols - 1) // cols

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"Room: {room}" for room in room_data.keys()],
            specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)],
        )

        for idx, (room_name, data) in enumerate(room_data.items()):
            row = idx // cols + 1
            col = idx % cols + 1

            # Temperature trace
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["temperature"],
                    name=f"{room_name} Temp",
                    line=dict(width=2),
                ),
                row=row,
                col=col,
            )

            # Relay state (secondary y-axis)
            if "relay_state" in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data["relay_state"],
                        name=f"{room_name} Relay",
                        line=dict(color="red", width=1),
                        yaxis="y2",
                    ),
                    row=row,
                    col=col,
                    secondary_y=True,
                )

            # Add RC parameters as annotation
            if room_name in rc_parameters:
                params = rc_parameters[room_name]
                annotation_text = (
                    f"R: {params.get('R', 0):.2f} K/kW<br>"
                    f"C: {params.get('C', 0):.0f} kJ/K<br>"
                    f"τ: {params.get('time_constant', 0):.1f} h"
                )

                fig.add_annotation(
                    text=annotation_text,
                    xref=f"x{idx+1}",
                    yref=f"y{idx+1}",
                    x=0.05,
                    y=0.95,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    row=row,
                    col=col,
                )

        # Update layout
        fig.update_layout(
            title="Thermal Analysis - RC Parameters for Relay-Controlled Rooms",
            height=300 * rows,
            showlegend=True,
            template="plotly_white",
        )

        # Update y-axis labels
        for i in range(1, num_rooms + 1):
            fig.update_yaxes(
                title_text="Temperature (°C)",
                row=(i - 1) // cols + 1,
                col=(i - 1) % cols + 1,
            )
            fig.update_yaxes(
                title_text="Relay State",
                secondary_y=True,
                row=(i - 1) // cols + 1,
                col=(i - 1) % cols + 1,
            )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_relay_optimization_dashboard(
        self,
        relay_data: pd.DataFrame,
        optimization_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create relay optimization analysis dashboard.

        Shows current vs optimized relay operation patterns.
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Peak Demand Analysis",
                "Relay Coordination Opportunities",
                "Daily Load Pattern Optimization",
                "Economic Impact",
            ),
        )

        # 1. Peak Demand Analysis
        daily_peak = relay_data.resample("D")["total_power"].max()
        fig.add_trace(
            go.Scatter(
                x=daily_peak.index,
                y=daily_peak.values,
                name="Current Daily Peak (kW)",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )

        if "optimized_peak" in optimization_results:
            optimized_peak = optimization_results["optimized_peak"]
            fig.add_trace(
                go.Scatter(
                    x=daily_peak.index,
                    y=optimized_peak,
                    name="Optimized Daily Peak (kW)",
                    line=dict(color="green", width=2, dash="dash"),
                ),
                row=1,
                col=1,
            )

        # 2. Relay Coordination Opportunities
        if "coordination_matrix" in optimization_results:
            coordination_matrix = optimization_results["coordination_matrix"]

            fig.add_trace(
                go.Heatmap(
                    z=coordination_matrix.values,
                    x=coordination_matrix.columns,
                    y=coordination_matrix.index,
                    colorscale="RdYlBu_r",
                    name="Coordination Score",
                ),
                row=1,
                col=2,
            )

        # 3. Daily Load Pattern Optimization
        hourly_avg = relay_data.groupby(relay_data.index.hour)["total_power"].mean()
        fig.add_trace(
            go.Bar(
                x=hourly_avg.index,
                y=hourly_avg.values,
                name="Current Hourly Average (kW)",
                marker_color="lightblue",
            ),
            row=2,
            col=1,
        )

        if "optimized_hourly" in optimization_results:
            optimized_hourly = optimization_results["optimized_hourly"]
            fig.add_trace(
                go.Bar(
                    x=hourly_avg.index,
                    y=optimized_hourly,
                    name="Optimized Hourly Average (kW)",
                    marker_color="darkgreen",
                    opacity=0.7,
                ),
                row=2,
                col=1,
            )

        # 4. Economic Impact
        if "economic_impact" in optimization_results:
            econ_impact = optimization_results["economic_impact"]

            categories = list(econ_impact.keys())
            values = list(econ_impact.values())

            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    name="Annual Savings (CZK)",
                    marker_color="green",
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title="Relay Optimization Dashboard - 16-Room Binary Control System",
            height=800,
            showlegend=True,
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_pv_export_constraint_analysis(
        self,
        pv_data: pd.DataFrame,
        price_data: pd.DataFrame,
        export_policy_change_date: datetime,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Analyze PV export behavior considering system constraints.

        Compares pre-export (forced self-consumption) vs post-export (price-based) periods.
        """
        # Split data into periods
        pre_export = pv_data[pv_data.index < export_policy_change_date]
        post_export = pv_data[pv_data.index >= export_policy_change_date]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Curtailment Analysis (Pre-Export Period)",
                "Export vs Price Correlation (Post-Export)",
                "Self-Consumption Ratio Evolution",
                "Economic Opportunity Analysis",
            ),
        )

        # 1. Curtailment Analysis
        if "curtailment" in pre_export.columns:
            daily_curtailment = pre_export.resample("D")["curtailment"].sum()
            fig.add_trace(
                go.Bar(
                    x=daily_curtailment.index,
                    y=daily_curtailment.values,
                    name="Daily Curtailment (kWh)",
                    marker_color="orange",
                ),
                row=1,
                col=1,
            )

        # 2. Export vs Price Correlation
        if "ExportPower" in post_export.columns:
            # Align price data with PV data
            aligned_data = post_export.merge(
                price_data, left_index=True, right_index=True, how="inner"
            )

            fig.add_trace(
                go.Scatter(
                    x=aligned_data["price"],
                    y=aligned_data["ExportPower"],
                    mode="markers",
                    name="Export vs Price",
                    marker=dict(
                        color=aligned_data.index.hour,
                        colorscale="Viridis",
                        size=6,
                        colorbar=dict(title="Hour of Day"),
                    ),
                ),
                row=1,
                col=2,
            )

            # Add threshold line
            fig.add_vline(
                x=self.export_price_threshold,
                line=dict(color="red", dash="dash"),
                annotation_text="Export Threshold",
                row=1,
                col=2,
            )

        # 3. Self-Consumption Ratio Evolution
        if "SelfConsumption" in pv_data.columns:
            daily_pv = pv_data.resample("D")["InputPower"].sum()
            daily_self = pv_data.resample("D")["SelfConsumption"].sum()
            self_consumption_ratio = (daily_self / daily_pv).fillna(0)

            fig.add_trace(
                go.Scatter(
                    x=self_consumption_ratio.index,
                    y=self_consumption_ratio * 100,
                    name="Self-Consumption Ratio (%)",
                    line=dict(color="blue", width=2),
                ),
                row=2,
                col=1,
            )

            # Add policy change vertical line
            fig.add_vline(
                x=export_policy_change_date,
                line=dict(color="red", dash="dash"),
                annotation_text="Export Policy Change",
                row=2,
                col=1,
            )

        # 4. Economic Opportunity Analysis
        # Calculate potential revenue from curtailed energy
        if "curtailment" in pre_export.columns:
            monthly_curtailment = pre_export.resample("M")["curtailment"].sum()

            # Estimate potential revenue at average market price
            avg_price = (
                price_data["price"].mean() if len(price_data) > 0 else 2.0
            )  # CZK/kWh fallback
            potential_revenue = monthly_curtailment * avg_price / 1000  # Convert to CZK

            fig.add_trace(
                go.Bar(
                    x=potential_revenue.index,
                    y=potential_revenue.values,
                    name="Lost Revenue (CZK/month)",
                    marker_color="red",
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title="PV Export Constraint Analysis - Policy Impact Assessment",
            height=800,
            showlegend=True,
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def save_static_plots(
        self,
        data: Dict[str, pd.DataFrame],
        analysis_results: Dict[str, Any],
        prefix: str = "analysis",
    ) -> List[str]:
        """
        Save static matplotlib plots for reports.

        Returns list of saved file paths.
        """
        saved_files = []

        # PV Production Summary
        if "pv" in data:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("PV System Analysis Summary", fontsize=16)

            pv_data = data["pv"]

            # Daily production
            daily_prod = pv_data.resample("D")["InputPower"].sum()
            axes[0, 0].plot(daily_prod.index, daily_prod.values)
            axes[0, 0].set_title("Daily PV Production")
            axes[0, 0].set_ylabel("Energy (kWh)")

            # Production distribution
            axes[0, 1].hist(pv_data["InputPower"], bins=50, alpha=0.7)
            axes[0, 1].set_title("Production Distribution")
            axes[0, 1].set_xlabel("Power (kW)")

            # Monthly averages
            monthly_avg = pv_data.resample("M")["InputPower"].mean()
            axes[1, 0].bar(range(len(monthly_avg)), monthly_avg.values)
            axes[1, 0].set_title("Monthly Average Production")
            axes[1, 0].set_ylabel("Power (kW)")

            # Self-consumption if available
            if "SelfConsumption" in pv_data.columns:
                daily_self = pv_data.resample("D")["SelfConsumption"].sum()
                ratio = (daily_self / daily_prod).fillna(0)
                axes[1, 1].plot(ratio.index, ratio * 100)
                axes[1, 1].set_title("Self-Consumption Ratio")
                axes[1, 1].set_ylabel("Percentage (%)")

            plt.tight_layout()
            pv_path = self.output_dir / f"{prefix}_pv_summary.png"
            plt.savefig(pv_path, dpi=300, bbox_inches="tight")
            plt.close()
            saved_files.append(str(pv_path))

        # Thermal Analysis Summary
        if "thermal_analysis" in analysis_results:
            thermal_data = analysis_results["thermal_analysis"]

            if "rc_parameters" in thermal_data:
                rc_params = thermal_data["rc_parameters"]

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle("Thermal Analysis - RC Parameters", fontsize=16)

                rooms = list(rc_params.keys())
                r_values = [rc_params[room].get("R", 0) for room in rooms]
                c_values = [rc_params[room].get("C", 0) for room in rooms]
                tau_values = [rc_params[room].get("time_constant", 0) for room in rooms]

                # Thermal resistance
                axes[0].bar(rooms, r_values)
                axes[0].set_title("Thermal Resistance (R)")
                axes[0].set_ylabel("K/kW")
                axes[0].tick_params(axis="x", rotation=45)

                # Thermal capacitance
                axes[1].bar(rooms, c_values)
                axes[1].set_title("Thermal Capacitance (C)")
                axes[1].set_ylabel("kJ/K")
                axes[1].tick_params(axis="x", rotation=45)

                # Time constant
                axes[2].bar(rooms, tau_values)
                axes[2].set_title("Time Constant (τ)")
                axes[2].set_ylabel("Hours")
                axes[2].tick_params(axis="x", rotation=45)

                plt.tight_layout()
                thermal_path = self.output_dir / f"{prefix}_thermal_parameters.png"
                plt.savefig(thermal_path, dpi=300, bbox_inches="tight")
                plt.close()
                saved_files.append(str(thermal_path))

        return saved_files

    def plot_relay_patterns(
        self,
        relay_data: pd.DataFrame,
        analysis_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create relay pattern visualization.

        Required visualizations:
        1. Relay state timeline for each room
        2. Duty cycle comparison bar chart
        3. Energy consumption by room (stacked area)
        4. Switching frequency heatmap
        """
        if relay_data.empty:
            raise ValueError("No relay data provided")

        # Determine number of rooms and create appropriate subplot layout
        rooms = [
            col
            for col in relay_data.columns
            if "relay" in col.lower() or "heating" in col.lower()
        ]
        n_rooms = len(rooms)

        if n_rooms == 0:
            raise ValueError("No relay columns found in data")

        # Create subplot layout
        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=(
                "Relay State Timeline (Sample Period)",
                "Duty Cycle by Room",
                "Energy Consumption by Room",
                "Daily Energy Pattern",
                "Switching Frequency by Hour",
                "Peak Demand Analysis",
                "Room Correlation Matrix",
                "Weekly Usage Patterns",
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}],
            ],
            vertical_spacing=0.08,
        )

        # 1. Relay State Timeline (sample last 7 days)
        sample_data = relay_data.tail(
            7 * 24 * 12
        )  # Last 7 days assuming 5-min intervals
        y_position = 0
        colors = px.colors.qualitative.Set3

        for i, room in enumerate(rooms[:10]):  # Limit to 10 rooms for readability
            if room in sample_data.columns:
                room_states = sample_data[room]
                # Create blocks for ON periods
                on_periods = room_states[room_states > 0]

                if not on_periods.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=on_periods.index,
                            y=[y_position] * len(on_periods),
                            mode="markers",
                            marker=dict(
                                symbol="square", size=8, color=colors[i % len(colors)]
                            ),
                            name=room.replace("_", " ").title(),
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )
            y_position += 1

        fig.update_yaxes(
            tickvals=list(range(len(rooms[:10]))),
            ticktext=[room.replace("_", " ").title() for room in rooms[:10]],
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Time", row=1, col=1)

        # 2. Duty Cycle by Room
        duty_cycles = {}
        for room in rooms:
            if room in relay_data.columns:
                duty_cycle = (relay_data[room] > 0).mean() * 100
                duty_cycles[room] = duty_cycle

        sorted_rooms = sorted(duty_cycles.items(), key=lambda x: x[1], reverse=True)
        room_names = [room.replace("_", " ").title() for room, _ in sorted_rooms[:12]]
        duty_values = [duty for _, duty in sorted_rooms[:12]]

        fig.add_trace(
            go.Bar(
                x=room_names,
                y=duty_values,
                name="Duty Cycle (%)",
                marker_color="lightblue",
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(tickangle=45, row=2, col=1)
        fig.update_yaxes(title_text="Duty Cycle (%)", row=2, col=1)

        # Use power ratings from configuration
        from config.energy_settings import get_room_power

        energy_consumption = {}
        for room in rooms:
            if room in relay_data.columns:
                # Extract room name from column
                room_name = room.lower().replace("relay_", "").replace("_heating", "")
                power_rating = get_room_power(room_name)  # Get from config

                # Calculate energy (assuming 5-min intervals)
                energy_kwh = (
                    (relay_data[room] > 0).sum() * power_rating * (5 / 60) / 1000
                )
                energy_consumption[room] = energy_kwh

        sorted_energy = sorted(
            energy_consumption.items(), key=lambda x: x[1], reverse=True
        )
        energy_names = [
            room.replace("_", " ").title() for room, _ in sorted_energy[:12]
        ]
        energy_values = [energy for _, energy in sorted_energy[:12]]

        fig.add_trace(
            go.Bar(
                x=energy_names,
                y=energy_values,
                name="Energy (kWh)",
                marker_color="orange",
            ),
            row=2,
            col=2,
        )
        fig.update_xaxes(tickangle=45, row=2, col=2)
        fig.update_yaxes(title_text="Energy Consumption (kWh)", row=2, col=2)

        # 4. Daily Energy Pattern
        if len(relay_data) > 24 * 12:  # At least 1 day of data
            total_power = pd.Series(0, index=relay_data.index)
            for room in rooms:
                if room in relay_data.columns:
                    room_name = (
                        room.lower().replace("relay_", "").replace("_heating", "")
                    )
                    power_rating = get_room_power(room_name)
                    total_power += (relay_data[room] > 0) * power_rating

            hourly_avg = total_power.groupby(total_power.index.hour).mean()

            fig.add_trace(
                go.Scatter(
                    x=hourly_avg.index,
                    y=hourly_avg.values,
                    mode="lines+markers",
                    name="Average Hourly Power",
                    line=dict(color="green", width=3),
                ),
                row=3,
                col=1,
            )
            fig.update_xaxes(title_text="Hour of Day", row=3, col=1)
            fig.update_yaxes(title_text="Power (kW)", row=3, col=1)

        # 5. Peak Demand Analysis
        if len(relay_data) > 24 * 12:
            daily_peak = total_power.resample("D").max()

            fig.add_trace(
                go.Scatter(
                    x=daily_peak.index,
                    y=daily_peak.values,
                    mode="lines+markers",
                    name="Daily Peak Demand",
                    line=dict(color="red", width=2),
                ),
                row=3,
                col=2,
            )
            fig.update_xaxes(title_text="Date", row=3, col=2)
            fig.update_yaxes(title_text="Peak Power (kW)", row=3, col=2)

        # 6. Switching Frequency Heatmap
        switching_matrix = np.zeros((24, 7))  # Hours x Days of week

        for room in rooms[:5]:  # Limit to avoid overcrowding
            if room in relay_data.columns:
                switches = relay_data[room].diff().abs() > 0
                switch_times = relay_data[switches].index

                for switch_time in switch_times:
                    hour = switch_time.hour
                    day = switch_time.weekday()
                    switching_matrix[hour, day] += 1

        fig.add_trace(
            go.Heatmap(
                z=switching_matrix,
                x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                y=list(range(24)),
                colorscale="Reds",
                name="Switches",
            ),
            row=4,
            col=1,
        )
        fig.update_yaxes(title_text="Hour of Day", row=4, col=1)

        # 7. Weekly Usage Pattern
        if len(relay_data) > 7 * 24 * 12:
            weekly_usage = {}
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

            for day_idx in range(7):
                day_data = relay_data[relay_data.index.weekday == day_idx]
                if not day_data.empty:
                    avg_active_relays = sum(
                        (day_data[room] > 0).mean()
                        for room in rooms
                        if room in day_data.columns
                    )
                    weekly_usage[days[day_idx]] = avg_active_relays

            fig.add_trace(
                go.Bar(
                    x=list(weekly_usage.keys()),
                    y=list(weekly_usage.values()),
                    name="Avg Active Relays",
                    marker_color="purple",
                ),
                row=4,
                col=2,
            )
            fig.update_yaxes(title_text="Average Active Relays", row=4, col=2)

        # Update layout
        fig.update_layout(
            title="Relay System Analysis Dashboard",
            height=1400,
            showlegend=False,
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_base_load_analysis(
        self,
        base_load_data: pd.DataFrame,
        analysis_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create base load analysis dashboard.

        Required plots:
        1. Daily load profiles (weekday vs weekend)
        2. Load duration curve
        3. Clustering results visualization
        4. Anomaly detection results
        """
        if base_load_data.empty:
            raise ValueError("No base load data provided")

        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Daily Load Profiles (Weekday vs Weekend)",
                "Load Duration Curve",
                "Monthly Load Patterns",
                "Load Clustering Results",
                "Anomaly Detection",
                "Load Statistics Summary",
            ),
        )

        # Assume base_load_data has a 'load' column
        load_col = (
            "load" if "load" in base_load_data.columns else base_load_data.columns[0]
        )
        load_data = base_load_data[load_col]

        # 1. Daily Load Profiles
        weekday_profile = (
            load_data[load_data.index.weekday < 5].groupby(load_data.index.hour).mean()
        )
        weekend_profile = (
            load_data[load_data.index.weekday >= 5].groupby(load_data.index.hour).mean()
        )

        fig.add_trace(
            go.Scatter(
                x=weekday_profile.index,
                y=weekday_profile.values,
                mode="lines+markers",
                name="Weekday Average",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=weekend_profile.index,
                y=weekend_profile.values,
                mode="lines+markers",
                name="Weekend Average",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )

        fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
        fig.update_yaxes(title_text="Load (kW)", row=1, col=1)

        # 2. Load Duration Curve
        sorted_load = load_data.sort_values(ascending=False).reset_index(drop=True)
        percentiles = np.arange(len(sorted_load)) / len(sorted_load) * 100

        fig.add_trace(
            go.Scatter(
                x=percentiles,
                y=sorted_load.values,
                mode="lines",
                name="Load Duration Curve",
                line=dict(color="green", width=2),
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Percent of Time", row=1, col=2)
        fig.update_yaxes(title_text="Load (kW)", row=1, col=2)

        # 3. Monthly Load Patterns
        if len(load_data) > 30 * 24 * 12:  # At least 30 days
            monthly_avg = load_data.resample("M").mean()
            monthly_max = load_data.resample("M").max()

            fig.add_trace(
                go.Scatter(
                    x=monthly_avg.index,
                    y=monthly_avg.values,
                    mode="lines+markers",
                    name="Monthly Average",
                    line=dict(color="blue"),
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=monthly_max.index,
                    y=monthly_max.values,
                    mode="lines+markers",
                    name="Monthly Peak",
                    line=dict(color="red"),
                ),
                row=2,
                col=1,
            )

            fig.update_xaxes(title_text="Month", row=2, col=1)
            fig.update_yaxes(title_text="Load (kW)", row=2, col=1)

        # 4. Load Clustering Results (if available in analysis_results)
        if (
            "clustering" in analysis_results
            and "labels" in analysis_results["clustering"]
        ):
            cluster_labels = analysis_results["clustering"]["labels"]

            # Create hourly load vs cluster scatter
            if len(cluster_labels) == len(load_data):
                hourly_load = load_data.groupby(load_data.index.hour).mean()
                cluster_counts = pd.Series(cluster_labels).value_counts()

                fig.add_trace(
                    go.Bar(
                        x=[f"Cluster {i}" for i in cluster_counts.index],
                        y=cluster_counts.values,
                        name="Cluster Sizes",
                        marker_color="lightblue",
                    ),
                    row=2,
                    col=2,
                )

                fig.update_xaxes(title_text="Cluster", row=2, col=2)
                fig.update_yaxes(title_text="Number of Days", row=2, col=2)

        # 5. Anomaly Detection (if available)
        if "anomalies" in analysis_results:
            anomaly_indices = analysis_results["anomalies"].get("anomaly_dates", [])

            if anomaly_indices:
                # Plot load with anomalies highlighted
                normal_data = load_data.drop(anomaly_indices, errors="ignore")
                anomaly_data = (
                    load_data.loc[anomaly_indices] if anomaly_indices else pd.Series()
                )

                # Sample data for visualization
                sample_data = load_data.tail(7 * 24 * 12)  # Last 7 days

                fig.add_trace(
                    go.Scatter(
                        x=sample_data.index,
                        y=sample_data.values,
                        mode="lines",
                        name="Normal Load",
                        line=dict(color="blue"),
                    ),
                    row=3,
                    col=1,
                )

                # Highlight anomalies in sample period
                sample_anomalies = [
                    idx for idx in anomaly_indices if idx in sample_data.index
                ]
                if sample_anomalies:
                    fig.add_trace(
                        go.Scatter(
                            x=sample_anomalies,
                            y=[sample_data.loc[idx] for idx in sample_anomalies],
                            mode="markers",
                            name="Anomalies",
                            marker=dict(color="red", size=8),
                        ),
                        row=3,
                        col=1,
                    )

                fig.update_xaxes(title_text="Time", row=3, col=1)
                fig.update_yaxes(title_text="Load (kW)", row=3, col=1)

        # 6. Load Statistics Summary
        stats = {
            "Mean": load_data.mean(),
            "Max": load_data.max(),
            "Min": load_data.min(),
            "Std": load_data.std(),
            "95th Percentile": load_data.quantile(0.95),
            "Load Factor": load_data.mean() / load_data.max()
            if load_data.max() > 0
            else 0,
        }

        fig.add_trace(
            go.Bar(
                x=list(stats.keys()),
                y=list(stats.values()),
                name="Load Statistics",
                marker_color="lightgreen",
            ),
            row=3,
            col=2,
        )

        fig.update_xaxes(tickangle=45, row=3, col=2)
        fig.update_yaxes(title_text="Value", row=3, col=2)

        # Update layout
        fig.update_layout(
            title="Base Load Analysis Dashboard",
            height=1200,
            showlegend=True,
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_analysis_summary_report(
        self, analysis_results: Dict[str, Any], save_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive analysis summary report.

        Returns HTML report with key findings and recommendations.
        """
        from datetime import datetime

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PEMS v2 Analysis Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                h3 {{ color: #7f8c8d; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                .recommendation {{ background-color: #e8f5e8; padding: 15px; margin: 10px 0; border-left: 4px solid #27ae60; border-radius: 0 5px 5px 0; }}
                .warning {{ background-color: #fdf2e9; padding: 15px; margin: 10px 0; border-left: 4px solid #e67e22; border-radius: 0 5px 5px 0; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; text-align: center; color: #7f8c8d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏠 PEMS v2 Analysis Summary Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>📊 Executive Summary</h2>
        """

        # System Overview
        if "system_overview" in analysis_results:
            overview = analysis_results["system_overview"]
            html_content += f"""
                <div class="grid">
                    <div class="metric">
                        <div class="metric-value">{overview.get('total_rooms', 'N/A')}</div>
                        <div class="metric-label">Total Rooms</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{overview.get('total_capacity_kw', 'N/A'):.1f} kW</div>
                        <div class="metric-label">Total Heating Capacity</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{overview.get('pv_capacity_kw', 'N/A'):.1f} kW</div>
                        <div class="metric-label">PV System Capacity</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{overview.get('data_completeness', 'N/A'):.1f}%</div>
                        <div class="metric-label">Data Completeness</div>
                    </div>
                </div>
            """

        # PV Analysis Results
        if "pv_analysis" in analysis_results:
            pv = analysis_results["pv_analysis"]
            html_content += f"""
                <h2>☀️ PV System Performance</h2>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-value">{pv.get('total_production_kwh', 0):.0f} kWh</div>
                        <div class="metric-label">Total Production</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{pv.get('self_consumption_ratio', 0):.1%}</div>
                        <div class="metric-label">Self-Consumption Ratio</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{pv.get('capacity_factor', 0):.1%}</div>
                        <div class="metric-label">Capacity Factor</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{pv.get('curtailment_kwh', 0):.0f} kWh</div>
                        <div class="metric-label">Curtailed Energy</div>
                    </div>
                </div>
            """

            # PV Recommendations
            if pv.get("curtailment_kwh", 0) > 1000:
                html_content += """
                    <div class="recommendation">
                        <strong>💡 PV Optimization Opportunity:</strong> Significant energy curtailment detected. 
                        Consider battery storage or load shifting to capture unused PV production.
                    </div>
                """

        # Heating System Analysis
        if "heating_analysis" in analysis_results:
            heating = analysis_results["heating_analysis"]
            html_content += f"""
                <h2>🔥 Heating System Performance</h2>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-value">{heating.get('peak_demand_kw', 0):.1f} kW</div>
                        <div class="metric-label">Peak Demand</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{heating.get('average_demand_kw', 0):.1f} kW</div>
                        <div class="metric-label">Average Demand</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{heating.get('load_factor', 0):.1%}</div>
                        <div class="metric-label">Load Factor</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{heating.get('efficiency_score', 0):.1f}/10</div>
                        <div class="metric-label">Efficiency Score</div>
                    </div>
                </div>
            """

            # Room Usage Table
            if "room_usage" in heating:
                html_content += """
                    <h3>Room Usage Analysis</h3>
                    <table>
                        <tr><th>Room</th><th>Usage (%)</th><th>Energy (kWh)</th><th>Efficiency</th></tr>
                """

                for room, stats in heating["room_usage"].items():
                    html_content += f"""
                        <tr>
                            <td>{room.replace('_', ' ').title()}</td>
                            <td>{stats.get('usage_percent', 0):.1f}%</td>
                            <td>{stats.get('energy_kwh', 0):.1f}</td>
                            <td>{stats.get('efficiency', 'N/A')}</td>
                        </tr>
                    """

                html_content += "</table>"

        # Optimization Recommendations
        html_content += """
            <h2>🎯 Optimization Recommendations</h2>
        """

        recommendations = []

        # Add recommendations based on analysis results
        if "pv_analysis" in analysis_results:
            pv = analysis_results["pv_analysis"]
            if pv.get("curtailment_kwh", 0) > 500:
                recommendations.append(
                    {
                        "priority": "HIGH",
                        "category": "PV System",
                        "title": "Reduce Energy Curtailment",
                        "description": f"Install battery storage or implement load shifting to capture {pv.get('curtailment_kwh', 0):.0f} kWh of curtailed energy annually.",
                        "potential_savings": f"{pv.get('curtailment_kwh', 0) * 6:.0f} CZK/year",
                    }
                )

        if "heating_analysis" in analysis_results:
            heating = analysis_results["heating_analysis"]
            if heating.get("load_factor", 1) < 0.5:
                recommendations.append(
                    {
                        "priority": "MEDIUM",
                        "category": "Heating System",
                        "title": "Improve Load Factor",
                        "description": "Implement relay coordination to reduce peak demand and improve system efficiency.",
                        "potential_savings": f"{heating.get('peak_demand_kw', 0) * 0.2 * 6000:.0f} CZK/year",
                    }
                )

        if "integration_opportunities" in analysis_results:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "System Integration",
                    "title": "PV-Heating Integration",
                    "description": "Shift heating loads to coincide with PV production periods to increase self-consumption.",
                    "potential_savings": "15-25% reduction in electricity costs",
                }
            )

        # Display recommendations
        for rec in recommendations:
            priority_color = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#95a5a6"}[
                rec["priority"]
            ]
            html_content += f"""
                <div class="recommendation" style="border-left-color: {priority_color};">
                    <h4 style="margin: 0 0 10px 0; color: {priority_color};">[{rec['priority']}] {rec['title']}</h4>
                    <p><strong>Category:</strong> {rec['category']}</p>
                    <p>{rec['description']}</p>
                    <p><strong>Potential Savings:</strong> {rec['potential_savings']}</p>
                </div>
            """

        # Technical Details
        html_content += """
            <h2>🔧 Technical Details</h2>
            <h3>Analysis Parameters</h3>
            <ul>
        """

        if "analysis_config" in analysis_results:
            config = analysis_results["analysis_config"]
            html_content += f"""
                <li><strong>Analysis Period:</strong> {config.get('start_date', 'N/A')} to {config.get('end_date', 'N/A')}</li>
                <li><strong>Data Points:</strong> {config.get('total_data_points', 'N/A'):,}</li>
                <li><strong>Missing Data:</strong> {config.get('missing_data_percent', 0):.1f}%</li>
                <li><strong>Analysis Version:</strong> PEMS v2.0</li>
            """

        html_content += """
            </ul>
            
            <h3>Data Quality Metrics</h3>
        """

        if "data_quality" in analysis_results:
            quality = analysis_results["data_quality"]

            # Data Quality Table
            html_content += """
                <table>
                    <tr><th>Data Source</th><th>Completeness</th><th>Quality Score</th><th>Issues</th></tr>
            """

            for source, metrics in quality.items():
                html_content += f"""
                    <tr>
                        <td>{source.replace('_', ' ').title()}</td>
                        <td>{metrics.get('completeness', 0):.1f}%</td>
                        <td>{metrics.get('quality_score', 0):.1f}/10</td>
                        <td>{metrics.get('issues', 'None')}</td>
                    </tr>
                """

            html_content += "</table>"

        # Next Steps
        html_content += f"""
            <h2>📋 Next Steps</h2>
            <ol>
                <li><strong>Immediate Actions (Week 1-2):</strong>
                    <ul>
                        <li>Review and validate key findings</li>
                        <li>Prioritize high-impact recommendations</li>
                        <li>Plan implementation timeline</li>
                    </ul>
                </li>
                <li><strong>Short-term Implementation (Month 1-2):</strong>
                    <ul>
                        <li>Implement basic optimization strategies</li>
                        <li>Set up enhanced monitoring</li>
                        <li>Begin Phase 2 planning</li>
                    </ul>
                </li>
                <li><strong>Long-term Development (Month 3-6):</strong>
                    <ul>
                        <li>Deploy advanced control algorithms</li>
                        <li>Integrate machine learning models</li>
                        <li>Continuous optimization and monitoring</li>
                    </ul>
                </li>
            </ol>
            
            <div class="footer">
                <p>Generated by PEMS v2 Analysis Engine | Report ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}</p>
                <p>For technical support or questions about this report, please refer to the PEMS v2 documentation.</p>
            </div>
            
            </div>
        </body>
        </html>
        """

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        return html_content

    def plot_weather_correlation_matrix(
        self,
        weather_data: pd.DataFrame,
        energy_data: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create weather correlation matrix visualization.

        Args:
            weather_data: Weather measurements DataFrame
            energy_data: Energy consumption/production DataFrame
            save_path: Optional path to save the figure

        Returns:
            Plotly figure with correlation matrix heatmap
        """
        # Combine weather and energy data
        combined_data = pd.merge(
            weather_data, energy_data, left_index=True, right_index=True, how="inner"
        )

        # Select numeric columns only
        numeric_data = combined_data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            raise ValueError("No numeric data available for correlation analysis")

        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale="RdBu",
                zmid=0,
                text=correlation_matrix.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="<b>%{x}</b><br><b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
            )
        )

        # Update layout
        fig.update_layout(
            title="Weather-Energy Correlation Matrix",
            xaxis_title="Variables",
            yaxis_title="Variables",
            width=800,
            height=800,
            template="plotly_white",
        )

        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_feature_importance_dashboard(
        self,
        feature_importance: Dict[str, float],
        model_performance: Dict[str, float],
        feature_categories: Optional[Dict[str, List[str]]] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create feature importance analysis dashboard.

        Args:
            feature_importance: Dictionary of feature names and importance scores
            model_performance: Dictionary of model performance metrics
            feature_categories: Optional categorization of features
            save_path: Optional path to save the figure

        Returns:
            Plotly figure with feature importance analysis
        """
        if not feature_importance:
            raise ValueError("No feature importance data provided")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Top 20 Features by Importance",
                "Feature Importance by Category",
                "Model Performance Metrics",
                "Feature Importance Distribution",
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "histogram"}],
            ],
        )

        # 1. Top 20 Features
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:20]
        feature_names = [f[0] for f in sorted_features]
        importance_values = [f[1] for f in sorted_features]

        fig.add_trace(
            go.Bar(
                y=feature_names[::-1],  # Reverse for horizontal bar chart
                x=importance_values[::-1],
                orientation="h",
                name="Feature Importance",
                marker_color="lightblue",
                text=[f"{val:.3f}" for val in importance_values[::-1]],
                textposition="inside",
            ),
            row=1,
            col=1,
        )

        fig.update_xaxes(title_text="Importance Score", row=1, col=1)
        fig.update_yaxes(title_text="Features", row=1, col=1)

        # 2. Feature Importance by Category
        if feature_categories:
            category_importance = {}
            for category, features in feature_categories.items():
                category_score = sum(
                    feature_importance.get(feature, 0) for feature in features
                )
                category_importance[category] = category_score

            categories = list(category_importance.keys())
            category_scores = list(category_importance.values())

            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=category_scores,
                    name="Category Importance",
                    marker_color="orange",
                    text=[f"{val:.2f}" for val in category_scores],
                    textposition="outside",
                ),
                row=1,
                col=2,
            )

            fig.update_xaxes(title_text="Feature Category", tickangle=45, row=1, col=2)
            fig.update_yaxes(title_text="Total Importance", row=1, col=2)

        # 3. Model Performance Metrics
        if model_performance:
            metric_names = list(model_performance.keys())
            metric_values = list(model_performance.values())

            colors = [
                "green"
                if "r2" in name.lower() or "accuracy" in name.lower()
                else "red"
                if "error" in name.lower() or "loss" in name.lower()
                else "blue"
                for name in metric_names
            ]

            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name="Model Performance",
                    marker_color=colors,
                    text=[f"{val:.3f}" for val in metric_values],
                    textposition="outside",
                ),
                row=2,
                col=1,
            )

            fig.update_xaxes(title_text="Metrics", tickangle=45, row=2, col=1)
            fig.update_yaxes(title_text="Score", row=2, col=1)

        # 4. Feature Importance Distribution
        fig.add_trace(
            go.Histogram(
                x=list(feature_importance.values()),
                nbinsx=30,
                name="Importance Distribution",
                marker_color="purple",
                opacity=0.7,
            ),
            row=2,
            col=2,
        )

        fig.update_xaxes(title_text="Importance Score", row=2, col=2)
        fig.update_yaxes(title_text="Number of Features", row=2, col=2)

        # Update overall layout
        fig.update_layout(
            title="Feature Importance Analysis Dashboard",
            height=800,
            showlegend=False,
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_seasonal_decomposition(
        self,
        time_series: pd.Series,
        decomposition_results: Dict[str, pd.Series],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create seasonal decomposition visualization.

        Args:
            time_series: Original time series data
            decomposition_results: Dictionary containing 'trend', 'seasonal', 'residual' components
            save_path: Optional path to save the figure

        Returns:
            Plotly figure with seasonal decomposition components
        """
        if time_series.empty:
            raise ValueError("No time series data provided")

        # Create subplots
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                "Original Time Series",
                "Trend Component",
                "Seasonal Component",
                "Residual Component",
            ),
            vertical_spacing=0.08,
        )

        # 1. Original time series
        fig.add_trace(
            go.Scatter(
                x=time_series.index,
                y=time_series.values,
                mode="lines",
                name="Original",
                line=dict(color="blue", width=1),
            ),
            row=1,
            col=1,
        )

        # 2. Trend component
        if "trend" in decomposition_results:
            trend = decomposition_results["trend"]
            fig.add_trace(
                go.Scatter(
                    x=trend.index,
                    y=trend.values,
                    mode="lines",
                    name="Trend",
                    line=dict(color="red", width=2),
                ),
                row=2,
                col=1,
            )

        # 3. Seasonal component
        if "seasonal" in decomposition_results:
            seasonal = decomposition_results["seasonal"]
            fig.add_trace(
                go.Scatter(
                    x=seasonal.index,
                    y=seasonal.values,
                    mode="lines",
                    name="Seasonal",
                    line=dict(color="green", width=1),
                ),
                row=3,
                col=1,
            )

        # 4. Residual component
        if "residual" in decomposition_results:
            residual = decomposition_results["residual"]
            fig.add_trace(
                go.Scatter(
                    x=residual.index,
                    y=residual.values,
                    mode="lines",
                    name="Residual",
                    line=dict(color="purple", width=1),
                ),
                row=4,
                col=1,
            )

        # Update layout
        fig.update_layout(
            title="Seasonal Decomposition Analysis",
            height=1000,
            showlegend=False,
            template="plotly_white",
        )

        # Update x-axis labels
        for i in range(1, 5):
            fig.update_xaxes(title_text="Time" if i == 4 else "", row=i, col=1)

        # Update y-axis labels
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonal", row=3, col=1)
        fig.update_yaxes(title_text="Residual", row=4, col=1)

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_energy_balance_dashboard(
        self,
        energy_data: Dict[str, pd.DataFrame],
        analysis_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create comprehensive energy balance dashboard.

        Args:
            energy_data: Dictionary containing energy production/consumption data
            analysis_results: Analysis results with balance calculations
            save_path: Optional path to save the figure

        Returns:
            Plotly figure with energy balance analysis
        """
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Energy Production vs Consumption",
                "Net Energy Balance",
                "Self-Consumption Analysis",
                "Grid Interaction Patterns",
                "Energy Storage Utilization",
                "Economic Impact Analysis",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"secondary_y": False}],
            ],
        )

        # Sample data for last 30 days for better visualization
        sample_period_days = 30

        # 1. Energy Production vs Consumption
        if "pv" in energy_data and "consumption" in energy_data:
            pv_data = energy_data["pv"].tail(
                sample_period_days * 24 * 4
            )  # 15-min intervals
            consumption_data = energy_data["consumption"].tail(
                sample_period_days * 24 * 4
            )

            # Daily aggregation
            daily_pv = pv_data.resample("D").sum() if not pv_data.empty else pd.Series()
            daily_consumption = (
                consumption_data.resample("D").sum()
                if not consumption_data.empty
                else pd.Series()
            )

            if not daily_pv.empty:
                pv_col = (
                    daily_pv.columns[0] if len(daily_pv.columns) > 0 else "production"
                )
                fig.add_trace(
                    go.Scatter(
                        x=daily_pv.index,
                        y=daily_pv[pv_col]
                        if pv_col in daily_pv.columns
                        else daily_pv.iloc[:, 0],
                        mode="lines+markers",
                        name="PV Production",
                        line=dict(color="gold", width=2),
                    ),
                    row=1,
                    col=1,
                )

            if not daily_consumption.empty:
                cons_col = (
                    daily_consumption.columns[0]
                    if len(daily_consumption.columns) > 0
                    else "consumption"
                )
                fig.add_trace(
                    go.Scatter(
                        x=daily_consumption.index,
                        y=daily_consumption[cons_col]
                        if cons_col in daily_consumption.columns
                        else daily_consumption.iloc[:, 0],
                        mode="lines+markers",
                        name="Consumption",
                        line=dict(color="blue", width=2),
                    ),
                    row=1,
                    col=1,
                )

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1)

        # 2. Net Energy Balance
        if "energy_balance" in analysis_results:
            balance_data = analysis_results["energy_balance"]

            if "daily_balance" in balance_data:
                balance_series = pd.Series(balance_data["daily_balance"])

                # Color code positive/negative balance
                colors = ["green" if x > 0 else "red" for x in balance_series.values]

                fig.add_trace(
                    go.Bar(
                        x=balance_series.index,
                        y=balance_series.values,
                        name="Net Balance",
                        marker_color=colors,
                    ),
                    row=1,
                    col=2,
                )

        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Net Energy (kWh)", row=1, col=2)

        # 3. Self-Consumption Analysis
        if "self_consumption" in analysis_results:
            sc_data = analysis_results["self_consumption"]

            if "monthly_ratio" in sc_data:
                monthly_ratio = pd.Series(sc_data["monthly_ratio"])

                fig.add_trace(
                    go.Scatter(
                        x=monthly_ratio.index,
                        y=monthly_ratio.values * 100,
                        mode="lines+markers",
                        name="Self-Consumption Ratio",
                        line=dict(color="green", width=3),
                    ),
                    row=2,
                    col=1,
                )

        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Self-Consumption (%)", row=2, col=1)

        # 4. Grid Interaction Patterns
        if "grid_interaction" in analysis_results:
            grid_data = analysis_results["grid_interaction"]

            # Hourly import/export pattern
            if "hourly_pattern" in grid_data:
                hourly_import = grid_data["hourly_pattern"].get("import", {})
                hourly_export = grid_data["hourly_pattern"].get("export", {})

                hours = list(range(24))
                import_values = [hourly_import.get(h, 0) for h in hours]
                export_values = [
                    -hourly_export.get(h, 0) for h in hours
                ]  # Negative for export

                fig.add_trace(
                    go.Bar(
                        x=hours,
                        y=import_values,
                        name="Grid Import",
                        marker_color="red",
                    ),
                    row=2,
                    col=2,
                )

                fig.add_trace(
                    go.Bar(
                        x=hours,
                        y=export_values,
                        name="Grid Export",
                        marker_color="green",
                    ),
                    row=2,
                    col=2,
                )

        fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
        fig.update_yaxes(title_text="Grid Power (kW)", row=2, col=2)

        # 5. Energy Storage Utilization
        if "battery" in energy_data and not energy_data["battery"].empty:
            battery_data = energy_data["battery"].tail(sample_period_days * 24 * 4)

            if "SOC" in battery_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=battery_data.index,
                        y=battery_data["SOC"],
                        mode="lines",
                        name="Battery SOC",
                        line=dict(color="blue", width=2),
                    ),
                    row=3,
                    col=1,
                )

            # Add charge/discharge power on secondary y-axis
            if "power" in battery_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=battery_data.index,
                        y=battery_data["power"],
                        mode="lines",
                        name="Battery Power",
                        line=dict(color="red", width=1),
                        yaxis="y2",
                    ),
                    row=3,
                    col=1,
                    secondary_y=True,
                )

        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="State of Charge (%)", row=3, col=1)

        # 6. Economic Impact Analysis
        if "economic_impact" in analysis_results:
            econ_data = analysis_results["economic_impact"]

            if "monthly_savings" in econ_data:
                savings_data = pd.Series(econ_data["monthly_savings"])

                fig.add_trace(
                    go.Bar(
                        x=savings_data.index,
                        y=savings_data.values,
                        name="Monthly Savings",
                        marker_color="darkgreen",
                        text=[f"{val:.0f} CZK" for val in savings_data.values],
                        textposition="outside",
                    ),
                    row=3,
                    col=2,
                )

        fig.update_xaxes(title_text="Month", row=3, col=2)
        fig.update_yaxes(title_text="Savings (CZK)", row=3, col=2)

        # Update overall layout
        fig.update_layout(
            title="Energy Balance Analysis Dashboard",
            height=1200,
            showlegend=True,
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)

        return fig
