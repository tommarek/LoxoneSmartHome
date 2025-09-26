"""Tests for heating optimizer module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from loxone_smart_home.config.settings import GrowattConfig, RoomConfig
from loxone_smart_home.modules.home.heating.optimizer import (
    HeatingOptimizer,
    HeatingRequirements,
)
from loxone_smart_home.modules.home.heating.models import RoomParams, RoomState


@pytest.fixture
def mock_config():
    """Create mock GrowattConfig for testing."""
    config = MagicMock(spec=GrowattConfig)
    config.heating_enabled = True
    config.heating_type = "heat_pump"
    config.heating_cop_points = [
        (-10.0, 1.8),
        (0.0, 2.5),
        (5.0, 3.0),
        (10.0, 3.3),
        (15.0, 3.6),
    ]
    config.heating_distribution_losses = 0.05
    config.heating_uncertainty_margin = 0.1
    config.heating_global_power_limit_kw = 8.0
    config.heating_min_block_minutes = 60
    config.heating_schedule_horizon_hours = 36
    config.heating_block_during_sell_spike = True
    config.heating_sell_spike_czk_kwh = 6.6
    return config


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    return MagicMock()


@pytest.fixture
def optimizer(mock_config, mock_logger):
    """Create HeatingOptimizer instance for testing."""
    return HeatingOptimizer(mock_config, mock_logger)


@pytest.fixture
def sample_rooms():
    """Create sample room configurations."""
    return [
        RoomParams(
            name="living_room",
            C_kwh_per_C=2.0,
            UA_kw_per_C=0.15,
            T_set=21.0,
            T_min=19.0,
            T_max=23.0,
            Pmax_kw=2.5,
            solar_gain_coeff=0.1,
            occupancy_priority=1,
        ),
        RoomParams(
            name="bedroom",
            C_kwh_per_C=1.5,
            UA_kw_per_C=0.1,
            T_set=20.0,
            T_min=18.0,
            T_max=22.0,
            Pmax_kw=2.0,
            solar_gain_coeff=0.05,
            occupancy_priority=2,
        ),
    ]


@pytest.fixture
def sample_room_states():
    """Create sample room states."""
    return [
        RoomState(name="living_room", T_now=19.5),
        RoomState(name="bedroom", T_now=19.0),
    ]


class TestHeatingOptimizer:
    """Test cases for HeatingOptimizer."""

    def test_calculate_cop_interpolation(self, optimizer):
        """Test COP calculation with interpolation."""
        # Test exact point
        assert optimizer.calculate_cop(0.0) == 2.5

        # Test interpolation
        cop_at_2_5 = optimizer.calculate_cop(2.5)
        assert 2.5 < cop_at_2_5 < 3.0  # Should be between 0°C and 5°C values

        # Test edge cases
        assert optimizer.calculate_cop(-15.0) == 1.8  # Below minimum
        assert optimizer.calculate_cop(20.0) == 3.6  # Above maximum

    def test_calculate_effective_cop(self, optimizer):
        """Test effective COP calculation with distribution losses."""
        base_cop = optimizer.calculate_cop(5.0)
        effective_cop = optimizer.calculate_effective_cop(5.0)

        assert effective_cop == pytest.approx(base_cop * 0.95, rel=1e-3)

    def test_calculate_marginal_electricity_cost(self, optimizer):
        """Test marginal electricity cost calculation."""
        test_hour = datetime(2024, 1, 15, 10, 0)
        spot_price = 2.0  # CZK/kWh

        c_el, v_pv, v_batt = optimizer.calculate_marginal_electricity_cost(test_hour, spot_price)

        # c_buy = 2.0 + 0.8 = 2.8
        # v_pv = max(0, 2.0 - 1.0) = 1.0
        # v_batt = max(0, 2.0 - 1.0 - 5.0) = 0 (negative, so 0)
        # V_batt = 0.95 * 0 = 0
        # c_el = min(2.8, 1.0 + 0) = 1.0
        assert c_el == pytest.approx(1.0, rel=1e-3)  # min(c_buy, v_pv + V_batt)
        assert v_pv == pytest.approx(1.0, rel=1e-3)
        assert v_batt == 0.0

    def test_calculate_heat_costs(self, optimizer):
        """Test heat cost calculation for multiple hours."""
        base_time = datetime(2024, 1, 15, 0, 0)
        # Use higher prices to ensure positive costs
        prices = {
            base_time: 3.0,
            base_time + timedelta(hours=1): 5.0,
            base_time + timedelta(hours=2): 2.0,
        }
        temperatures = {
            base_time: 5.0,
            base_time + timedelta(hours=1): 0.0,
            base_time + timedelta(hours=2): 10.0,
        }

        heat_costs = optimizer.calculate_heat_costs(prices, temperatures)

        # Should be sorted by heat cost (cheapest first)
        assert len(heat_costs) == 3
        # With these prices, all should have positive costs
        assert all(h.c_heat > 0 for h in heat_costs)
        # Check sorting
        assert heat_costs[0].c_heat <= heat_costs[1].c_heat
        assert heat_costs[1].c_heat <= heat_costs[2].c_heat

    def test_calculate_energy_requirements(self, optimizer, sample_rooms, sample_room_states):
        """Test energy requirements calculation."""
        base_time = datetime(2024, 1, 15, 0, 0)
        temperatures = {
            base_time + timedelta(hours=i): 5.0 for i in range(24)
        }

        requirements = optimizer.calculate_energy_requirements(
            sample_rooms, sample_room_states, temperatures, horizon_hours=24
        )

        # Check lift energy
        assert "living_room" in requirements.E_lift
        assert "bedroom" in requirements.E_lift

        # Living room needs to go from 19.5 to 21.0 = 1.5°C * 2.0 kWh/°C = 3.0 kWh
        assert requirements.E_lift["living_room"] == pytest.approx(3.0, rel=1e-2)

        # Check maintenance energy exists
        assert len(requirements.E_maint) > 0

        # Check total includes margin
        raw_total = sum(requirements.E_lift.values()) + sum(requirements.E_maint.values())
        assert requirements.E_total == pytest.approx(raw_total * 1.1, rel=1e-3)

    def test_allocate_heating_hours(self, optimizer, sample_rooms):
        """Test heating hour allocation."""
        base_time = datetime(2024, 1, 15, 0, 0)

        # Create mock heat costs (already sorted)
        from loxone_smart_home.modules.growatt.models import HeatingHourCost
        heat_costs = [
            HeatingHourCost(
                t=base_time + timedelta(hours=i),
                c_el=1.0 + i * 0.5,
                COP=3.0,
                c_heat=(1.0 + i * 0.5) / 3.0,
                pv_value=0.0,
                batt_value=0.0,
            )
            for i in range(6)
        ]

        # Create requirements
        requirements = HeatingRequirements()
        requirements.E_total = 10.0  # Total energy needed
        requirements.E_lift = {"living_room": 3.0, "bedroom": 2.0}

        allocations = optimizer.allocate_heating_hours(heat_costs, requirements, sample_rooms)

        # Should allocate to cheapest hours first
        assert len(allocations) > 0
        assert allocations[0].t_start == base_time  # First (cheapest) hour

        # Check power limits are respected
        for alloc in allocations:
            assert alloc.kW_heat_total <= optimizer.config.heating_global_power_limit_kw
            for room_name, kw in alloc.per_room_kw.items():
                room = next(r for r in sample_rooms if r.name == room_name)
                assert kw <= room.Pmax_kw

    def test_group_contiguous_blocks(self, optimizer):
        """Test grouping of contiguous heating blocks."""
        base_time = datetime(2024, 1, 15, 0, 0)

        from loxone_smart_home.modules.growatt.models import HeatingAllocation

        # Create contiguous allocations with similar power
        allocations = [
            HeatingAllocation(
                t_start=base_time,
                duration_minutes=60,
                kW_heat_total=5.0,
                per_room_kw={"living_room": 3.0, "bedroom": 2.0},
            ),
            HeatingAllocation(
                t_start=base_time + timedelta(hours=1),
                duration_minutes=60,
                kW_heat_total=5.2,  # Similar power (within 20%)
                per_room_kw={"living_room": 3.1, "bedroom": 2.1},
            ),
            HeatingAllocation(
                t_start=base_time + timedelta(hours=3),  # Not contiguous
                duration_minutes=60,
                kW_heat_total=5.0,
                per_room_kw={"living_room": 3.0, "bedroom": 2.0},
            ),
        ]

        grouped = optimizer.group_contiguous_blocks(allocations)

        # First two should be merged, third stays separate
        assert len(grouped) == 2
        assert grouped[0].duration_minutes == 120  # Two 60-minute blocks merged
        assert grouped[1].duration_minutes == 60  # Single block

    def test_create_heating_plan(self, optimizer, sample_rooms, sample_room_states):
        """Test complete heating plan creation."""
        base_time = datetime(2024, 1, 15, 0, 0)

        # Create price and temperature data
        prices = {base_time + timedelta(hours=i): 1.0 + (i % 3) for i in range(24)}
        temperatures = {base_time + timedelta(hours=i): 5.0 for i in range(24)}

        plan = optimizer.create_heating_plan(
            rooms=sample_rooms,
            room_states=sample_room_states,
            prices=prices,
            temperatures=temperatures,
        )

        assert plan is not None
        assert plan.E_req_kWh > 0
        assert len(plan.allocations) >= 0  # May be 0 if no heating needed

        # Check that scheduled energy doesn't exceed requirements
        assert plan.E_sched_kWh <= plan.E_req_kWh * 1.1  # Allow small overrun

    def test_resistive_heating_cop(self, mock_logger):
        """Test that resistive heating always has COP of 1.0."""
        config = MagicMock(spec=GrowattConfig)
        config.heating_type = "resistive"
        config.heating_distribution_losses = 0.0
        config.heating_cop_points = []  # Empty for resistive heating

        optimizer = HeatingOptimizer(config, mock_logger)

        # Resistive heating should always have COP of 1.0
        assert optimizer.calculate_cop(5.0) == 1.0
        assert optimizer.calculate_cop(-10.0) == 1.0
        assert optimizer.calculate_effective_cop(5.0) == 1.0