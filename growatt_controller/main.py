# main.py

import schedule
import datetime
import time
import logging
import sys
from pv_output import get_forecasted_power
from energy_prices import (
    fetch_best_available_prices,
    find_cheapest_x_consecutive_hours,
    find_n_cheapest_hours,  # Import the new function
)
from battery_control import (
    configure_battery_first_with_ac_charge,
    configure_battery_first_without_ac_charge,  # Import the new function
    disable_battery_first,
    connect_mqtt,
    disconnect_mqtt,
    set_simulate_mode,
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


def get_individual_hours_from_window(start_time, stop_time):
    """
    Given a start and stop time, return a list of individual (start, stop) hour tuples.
    For example, ('14:00', '16:00') returns [('14:00', '15:00'), ('15:00', '16:00')].
    """
    start_dt = datetime.datetime.strptime(start_time, "%H:%M")
    stop_dt = datetime.datetime.strptime(stop_time, "%H:%M")
    individual_hours = []
    current = start_dt
    while current < stop_dt:
        next_hour = current + datetime.timedelta(hours=1)
        individual_hours.append(
            (current.strftime("%H:%M"), next_hour.strftime("%H:%M"))
        )
        current = next_hour
    return individual_hours


def safe_configure_battery_first_with_ac_charge(start_time, stop_time):
    """
    Wrapper for configure_battery_first_with_ac_charge with exception handling.
    """
    try:
        configure_battery_first_with_ac_charge(start_time, stop_time)
    except Exception as e:
        logging.exception(
            "Error in configure_battery_first_with_ac_charge. Disabling battery-first mode."
        )
        disable_battery_first()


def safe_configure_battery_first_without_ac_charge(start_time, stop_time):
    """
    Wrapper for configure_battery_first_without_ac_charge with exception handling.
    """
    try:
        configure_battery_first_without_ac_charge(start_time, stop_time)
    except Exception as e:
        logging.exception(
            "Error in configure_battery_first_without_ac_charge. Disabling battery-first mode."
        )
        disable_battery_first()


def calculate_and_schedule_next_day():
    """Fetches energy prices, calculates the cheapest hours, and schedules tasks."""
    # Clear previously scheduled tasks related to battery-first and disabling (without clearing the recalculate task)
    schedule.clear("battery_first")
    schedule.clear("disable_battery_first")
    schedule.clear("disable_charging")

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
        disable_battery_first()
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

    # Find the 2 cheapest consecutive hours
    cheapest_consecutive = find_cheapest_x_consecutive_hours(hourly_prices, 2)
    if not cheapest_consecutive:
        logging.warning("No cheapest 2-hour consecutive window found.")
    else:
        logging.info(f"Cheapest consecutive 2-hour window: {cheapest_consecutive}")

    # Initialize a set to keep track of excluded hours
    excluded_hours = set()

    # If a cheapest consecutive window is found, determine its individual hours and exclude them
    if cheapest_consecutive:
        start_time, stop_time, avg_price = cheapest_consecutive
        individual_hours = get_individual_hours_from_window(start_time, stop_time)
        logging.info(f"Excluding individual hours: {individual_hours}")
        for hour in individual_hours:
            excluded_hours.add(hour)
            # Remove these hours from hourly_prices to exclude them from the 8 cheapest
            if hour in hourly_prices:
                del hourly_prices[hour]

    # Find the 8 cheapest individual hours from the remaining hours
    cheapest_hours = find_n_cheapest_hours(hourly_prices, n=8)
    logging.info(f"8 Cheapest hours (excluding consecutive hours): {cheapest_hours}")

    # Schedule disabling charging during the 8 cheapest hours
    for start_time, stop_time, price in cheapest_hours:
        logging.info(
            f"Scheduling disable charging but keep battery-first from {start_time} to {stop_time}"
        )
        schedule.every().day.at(start_time).do(
            safe_configure_battery_first_without_ac_charge, start_time, stop_time
        ).tag("disable_charging")

    # Schedule enabling battery-first + AC charge during the 2 cheapest consecutive hours
    if cheapest_consecutive:
        start_time, stop_time, avg_price = cheapest_consecutive
        logging.info(
            f"Scheduling battery-first mode with AC charge from {start_time} to {stop_time}"
        )
        schedule.every().day.at(start_time).do(
            safe_configure_battery_first_with_ac_charge, start_time, stop_time
        ).tag("battery_first")


def schedule_daily_calculation():
    """Schedules the calculation and scheduling of battery control at 23:59 every day."""
    # Schedule the first calculation immediately
    calculate_and_schedule_next_day()

    # This job remains in place and won't be cleared by clear tags
    logging.info("Scheduling daily calculation at 23:59.")
    schedule.every().day.at("23:59").do(calculate_and_schedule_next_day).tag(
        "recalculate"
    )

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logging.exception("An error occurred during schedule.run_pending.")
            disable_battery_first()
        time.sleep(60)  # Sleep for a minute to check the next scheduled task


if __name__ == "__main__":
    # Check if we are in simulate mode by reading command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "simulate":
        logging.info("Running in SIMULATION mode.")
        set_simulate_mode(True)  # Set the global SIMULATE flag in battery_control.py

    logging.info("Connecting to MQTT broker.")
    # Connect to the MQTT broker before starting the schedule
    connect_mqtt()

    try:
        # Schedule the daily calculation and scheduling of battery control
        schedule_daily_calculation()
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Shutting down.")
    except Exception as e:
        logging.exception("An unexpected error occurred in the main execution.")
        disable_battery_first()
    finally:
        # Disconnect from the MQTT broker when done
        logging.info("Disconnecting from MQTT broker.")
        disconnect_mqtt()
