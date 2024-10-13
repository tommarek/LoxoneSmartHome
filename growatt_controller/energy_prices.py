import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging


def fetch_ida_prices(ida_session, date=None):
    if date is None:
        # Use today's date if not specified
        date = datetime.now().strftime("%Y-%m-%d")

    # Construct the URL for the given IDA session and date
    url = f"https://www.ote-cr.cz/en/short-term-markets/electricity/cross-border-flows-and-profile-data-ida?date={date}&ida_session=IDA{ida_session}"
    logging.info(f"Fetching IDA prices from: {url}")

    # Fetch the webpage content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Parse the prices from the table
    prices = {}
    table = soup.find(
        "table", class_="report_table"
    )  # Locate the table with the prices
    rows = table.find_all("tr")

    for row in rows[1:]:  # Skip the header row
        time_interval = row.find(
            "th"
        ).text.strip()  # First <th> contains the time interval
        price_td = row.find("td")  # First <td> contains the price
        if price_td:
            price = float(
                price_td.text.strip().replace(",", ".")
            )  # Convert price to float and handle commas
            start, stop = time_interval.split(
                "-"
            )  # Split the interval into start and stop times
            prices[(start, stop)] = price

    return prices


# Fetch energy prices data from OTE (DAM) with time intervals as tuples
def fetch_dam_energy_prices(date=None):
    """
    Fetch energy prices data from OTE (DAM) with time intervals as tuples.
    Optionally accepts a specific date; if no date is provided, defaults to the next day.
    """
    if date is None:
        # If no date is specified, default to the next day's date
        date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Construct the URL with the given date
    url = f"https://www.ote-cr.cz/en/short-term-markets/electricity/day-ahead-market/@@chart-data?report_date={date}"
    logging.info(f"Fetching energy prices for {date} from: {url}")

    # Fetch the data
    response = requests.get(url)
    data = response.json()["data"]["dataLine"][1]["point"]

    # Extract hourly prices and format with (start, stop) tuples
    hourly_prices = {}
    for point in data:
        hour = int(point["x"]) - 1
        price = float(point["y"])
        start_time = f"{hour:02d}:00"
        stop_time = f"{hour + 1:02d}:00"
        hourly_prices[(start_time, stop_time)] = price

    return hourly_prices


def fetch_best_available_prices(ida_session, date=None):
    """
    Attempts to fetch prices from IDA first. If unsuccessful, falls back to DAM prices.

    Parameters:
    - ida_session: The IDA session (e.g., '1', '2', '3').
    - date: The date for which to fetch the prices (optional, defaults to the next day).

    Returns:
    - A dictionary with the best available prices, using (start, stop) tuples as keys.
    """
    try:
        logging.info(f"Attempting to fetch IDA session {ida_session} prices for {date}")
        ida_prices = fetch_ida_prices(ida_session, date)
        if ida_prices:
            logging.info(f"Successfully fetched IDA session {ida_session} prices")
            return ida_prices
        else:
            logging.warning(
                f"No IDA prices available for session {ida_session} on {date}"
            )
    except Exception as e:
        logging.error(
            f"Failed to fetch IDA prices for session {ida_session} on {date}: {e}"
        )

    # Fall back to DAM if IDA is unavailable or fails
    try:
        logging.info(f"Falling back to DAM prices for {date}")
        dam_prices = fetch_dam_energy_prices(date)
        if dam_prices:
            logging.info(f"Successfully fetched DAM prices for {date}")
            return dam_prices
        else:
            logging.warning(f"No DAM prices available for {date}")
    except Exception as e:
        logging.error(f"Failed to fetch DAM prices for {date}: {e}")

    # If both fail, return an empty dictionary
    return {}


def find_cheapest_x_consecutive_hours(prices, x=2):
    """
    Finds the X consecutive hours with the lowest total price, handling both 15-minute and hourly intervals.
    Returns the start time, end time, and average price for the cheapest period.
    """
    intervals = list(prices.keys())
    num_intervals = len(intervals)

    # Determine if we are dealing with 15-minute intervals or hourly intervals
    interval_duration = (
        datetime.strptime(intervals[1][0], "%H:%M")
        - datetime.strptime(intervals[0][0], "%H:%M")
    ).seconds // 60

    if interval_duration == 15:
        intervals_per_hour = 4  # For 15-minute intervals
    elif interval_duration == 60:
        intervals_per_hour = 1  # For hourly intervals
    else:
        raise ValueError("Unknown interval duration")

    # Calculate the number of intervals that make up X hours
    intervals_needed = x * intervals_per_hour

    if num_intervals < intervals_needed:
        return []  # Not enough data to find X consecutive hours

    cheapest_window = None
    min_price_sum = float("inf")  # Set to infinity initially

    # Loop through prices to find the X hours with the lowest total price
    for i in range(num_intervals - intervals_needed + 1):
        price_sum = sum(prices[intervals[j]] for j in range(i, i + intervals_needed))
        avg_price = price_sum / intervals_needed

        if price_sum < min_price_sum:
            min_price_sum = price_sum
            start_time = intervals[i][0]  # Start time of the first interval
            end_time = intervals[i + intervals_needed - 1][
                1
            ]  # End time of the last interval
            cheapest_window = (start_time, end_time, avg_price)

    return cheapest_window
