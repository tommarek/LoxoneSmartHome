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
    find_n_cheapest_hours,
)
from battery_control import (
    configure_battery_first_with_ac_charge,
    configure_battery_first_without_ac_charge,
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


def group_contiguous_hours(hours):
    """
    Groups contiguous hours into continuous ranges.

    Parameters:
    - hours: List of tuples [(start_time, stop_time, price), ...]

    Returns:
    - List of tuples [(group_start_time, group_end_time), ...]
    """
    if not hours:
        return []

    # Sort hours by start_time
    sorted_hours = sorted(
        hours, key=lambda x: datetime.datetime.strptime(x[0], "%H:%M")
    )

    groups = []
    current_group_start = sorted_hours[0][0]
    current_group_end = sorted_hours[0][1]

    for hour in sorted_hours[1:]:
        previous_end = datetime.datetime.strptime(current_group_end, "%H:%M")
        current_start = datetime.datetime.strptime(hour[0], "%H:%M")

        if current_start == previous_end:
            # Extend the current group
            current_group_end = hour[1]
        else:
            # Close the current group and start a new one
            groups.append((current_group_start, current_group_end))
            current_group_start = hour[0]
            current_group_end = hour[1]

    # Append the last group
    groups.append((current_group_start, current_group_end))

    return groups


def safe_configure_battery_first_with_ac_charge(start_time, stop_time):
    """
    Wrapper for configure_battery_first_with_ac_charge with exception handling.
    """
    try:
        configure_battery_first_with_ac_charge(start_time, stop_time)
        logging.info(
            f"Enabled battery-first mode with AC charge from {start_time} to {stop_time}."
        )
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
        logging.info(
            f"Disabled charging but kept battery-first mode from {start_time} to {stop_time}."
        )
    except Exception as e:
        logging.exception(
            "Error in configure_battery_first_without_ac_charge. Disabling battery-first mode."
        )
        disable_battery_first()


def calculate_and_schedule_next_day():
    """Fetches energy prices, calculates the cheapest hours, and schedules tasks."""
    # Clear previously scheduled tasks related to battery-first and disabling (without clearing the recalculate task)
    schedule.clear("battery_first_ac_charge")
    schedule.clear("battery_first_no_ac_charge")

    # Fetch the best available prices (IDA2 first, fallback to DAM)
    hourly_prices = fetch_best_available_prices(
        ida_session="1",
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

    # Define the number of individual cheap hours and consecutive hours
    num_individual_hours = 8  # Number of individual cheapest hours
    num_consecutive_hours = 3  # Number of consecutive hours to enable

    # TODO: not working on synology for some reason, fix it in docker
    # # Fetch forecasted power
    # forecasted_power = get_forecasted_power()
    # logging.info(f"Forecasted power for tomorrow: {forecasted_power}")

    # # Check if forecasted power is below the threshold
    # if forecasted_power >= POWER_THRESHOLD:
    #     logging.info(
    #         f"Forecasted power is {forecasted_power}, which is above the threshold of {POWER_THRESHOLD}. Skipping battery-first scheduling."
    #     )
    #     return

    # Step 1: Find the cheapest individual hours
    cheapest_individual_hours = find_n_cheapest_hours(
        hourly_prices, n=num_individual_hours
    )
    logging.info(f"8 Cheapest individual hours: {cheapest_individual_hours}")

    # Step 2: Find the cheapest consecutive hours
    cheapest_consecutive = find_cheapest_x_consecutive_hours(
        hourly_prices, num_consecutive_hours
    )
    if not cheapest_consecutive:
        logging.warning("No cheapest 2-hour consecutive window found.")
    else:
        logging.info(f"Cheapest consecutive 2-hour window: {cheapest_consecutive}")

    # Step 3: Remove the consecutive hours from the disabling list
    if cheapest_consecutive:
        # Extract individual hours within the consecutive window
        start_time, stop_time, _ = cheapest_consecutive
        consecutive_individual_hours = get_individual_hours_from_window(
            start_time, stop_time
        )
        logging.info(
            f"Consecutive individual hours to exclude: {consecutive_individual_hours}"
        )

        # Remove these hours from the disabling list
        cheapest_individual_hours = [
            hour
            for hour in cheapest_individual_hours
            if (hour[0], hour[1]) not in consecutive_individual_hours
        ]
        logging.info(
            f"Cheapest individual hours after exclusion: {cheapest_individual_hours}"
        )
    else:
        logging.info("Proceeding without excluding any hours.")

    # Ensure we have exactly 6 disabling hours
    if cheapest_consecutive:
        num_disabling_hours = 6
    else:
        num_disabling_hours = (
            8  # If no consecutive window found, keep all 8 disabling hours
        )

    # Select the top `num_disabling_hours` disabling hours
    cheapest_disabling_hours = cheapest_individual_hours[:num_disabling_hours]
    logging.info(
        f"{num_disabling_hours} Cheapest disabling hours: {cheapest_disabling_hours}"
    )

    # Step 4: Group the disabling hours into contiguous ranges
    grouped_disabling_hours = group_contiguous_hours(cheapest_disabling_hours)
    logging.info(f"Grouped disabling hours: {grouped_disabling_hours}")

    # Step 5: Schedule the disabling actions
    for group_start, group_end in grouped_disabling_hours:
        logging.info(
            f"Scheduling disable charging but keep battery-first from {group_start} to {group_end}"
        )
        schedule.every().day.at(group_start).do(
            safe_configure_battery_first_without_ac_charge, group_start, group_end
        ).tag("battery_first_no_ac_charge")

    # Step 6: Schedule the enabling actions
    if cheapest_consecutive:
        start_time, stop_time, _ = cheapest_consecutive
        logging.info(
            f"Scheduling battery-first mode with AC charge from {start_time} to {stop_time}"
        )
        schedule.every().day.at(start_time).do(
            safe_configure_battery_first_with_ac_charge, start_time, stop_time
        ).tag("battery_first_ac_charge")


def schedule_daily_calculation():
    """Schedules the calculation and scheduling of battery control at 23:59 every day."""
    # Schedule the first calculation immediately
    calculate_and_schedule_next_day()

    # Schedule the daily recalculation at 23:59
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
    finally:
        disable_battery_first()
        # Disconnect from the MQTT broker when done
        logging.info("Disconnecting from MQTT broker.")
        disconnect_mqtt()
