import schedule
import datetime
import time
import logging
import sys
from pv_output import get_forecasted_power
from energy_prices import (
    fetch_best_available_prices,
    find_cheapest_x_consecutive_hours,
)
from battery_control import (
    configure_battery_first_with_ac_charge,
    disable_battery_first,
    connect_mqtt,
    disconnect_mqtt,
    set_simulate_mode,  # Import the function to control simulation mode
)

# Define the power threshold
POWER_THRESHOLD = 40000

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to console
    ],
)


def calculate_and_schedule_next_day():
    """Fetches energy prices, calculates the cheapest 3 consecutive hours, and schedules tasks."""
    # Clear previously scheduled tasks related to battery-first and disabling (without clearing the recalculate task)
    schedule.clear("battery_first")
    schedule.clear("disable_battery_first")

    # Fetch the best available prices (IDA2 first, fallback to DAM)
    hourly_prices = fetch_best_available_prices(
        ida_session="2",
        date=(datetime.datetime.now() + datetime.timedelta(days=1)).strftime(
            "%Y-%m-%d"
        ),
    )

    # If prices cannot be fetched, skip scheduling
    if not hourly_prices:
        logging.error(
            "Failed to retrieve energy prices. Skipping MQTT message scheduling."
        )
        return

    logging.info(f"Energy prices for tomorrow: {hourly_prices}")

    # Fetch forecasted power
    forecasted_power = get_forecasted_power()
    logging.info(f"Forecasted power for tomorrow: {forecasted_power}")

    # Check if forecasted power is below the threshold
    if forecasted_power >= POWER_THRESHOLD:
        logging.info(
            f"Forecasted power is {forecasted_power}, which is above the threshold of {POWER_THRESHOLD}. Skipping battery-first scheduling."
        )
        return

    # Find the cheapest 3 consecutive hours
    cheapest_window = find_cheapest_x_consecutive_hours(hourly_prices, 3)

    if not cheapest_window:
        logging.warning("No cheapest 3-hour window found.")
        return

    # Schedule battery-first mode at the start of the cheapest 3-hour window
    start_time = cheapest_window[0]
    stop_time = cheapest_window[1]

    logging.info(
        f"Scheduling battery-first mode with AC charge from {start_time} to {stop_time}"
    )
    schedule.every().day.at(start_time).do(
        configure_battery_first_with_ac_charge, start_time, stop_time
    ).tag("battery_first")

    # Schedule disabling battery-first mode at stop_time (after the 3-hour window ends)
    logging.info(f"Scheduling disabling of battery-first mode after {stop_time}")
    schedule.every().day.at(stop_time).do(disable_battery_first).tag(
        "disable_battery_first"
    )


def schedule_daily_calculation():
    """Schedules the calculation and scheduling of battery and grid modes at 23:55 every day."""
    # This job remains in place and won't be cleared by clear('battery_first') or clear('disable_battery_first')
    logging.info("Scheduling daily calculation at 23:55.")
    schedule.every().day.at("23:55").do(calculate_and_schedule_next_day).tag(
        "recalculate"
    )

    while True:
        schedule.run_pending()
        time.sleep(60)  # Sleep for a minute to check the next scheduled task


if __name__ == "__main__":
    # Check if we are in simulate mode by reading command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "simulate":
        logging.info("Running in SIMULATION mode.")
        set_simulate_mode(True)  # Set the global SIMULATE flag in battery_control.py

    logging.info("Connecting to MQTT broker.")
    # Connect to the MQTT broker before starting the schedule
    connect_mqtt()

    # Schedule the daily calculation and scheduling of battery control
    # schedule_daily_calculation()
    calculate_and_schedule_next_day()

    # Disconnect from the MQTT broker when done
    logging.info("Disconnecting from MQTT broker.")
    disconnect_mqtt()
