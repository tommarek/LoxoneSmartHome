# energy_prices.py

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
    hourly_prices = {}
    if response.json()["data"]["dataLine"]:
        data = response.json()["data"]["dataLine"][1]["point"]

        # Extract hourly prices and format with (start, stop) tuples
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
    # let's not do this for now :)
    # try:
    #     logging.info(f"Attempting to fetch IDA session {ida_session} prices for {date}")
    #     ida_prices = fetch_ida_prices(ida_session, date)
    #     if ida_prices:
    #         logging.info(f"Successfully fetched IDA session {ida_session} prices")
    #         return ida_prices
    #     else:
    #         logging.warning(
    #             f"No IDA prices available for session {ida_session} on {date}"
    #         )
    # except Exception as e:
    #     logging.error(
    #         f"Failed to fetch IDA prices for session {ida_session} on {date}: {e}"
    #     )

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
    Returns a list of tuples [(start_time, stop_time, price), ...] for the cheapest consecutive window.
    """
    intervals = list(prices.keys())
    num_intervals = len(intervals)

    if num_intervals < 2:
        raise ValueError("Not enough intervals to determine interval duration.")

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

    cheapest_window = []
    min_price_sum = float("inf")  # Set to infinity initially

    # Loop through prices to find the X hours with the lowest total price
    for i in range(num_intervals - intervals_needed + 1):
        window_intervals = intervals[i : i + intervals_needed]
        price_sum = sum(prices[interval] for interval in window_intervals)

        if price_sum < min_price_sum:
            min_price_sum = price_sum
            cheapest_window = [
                (interval[0], interval[1], prices[interval])
                for interval in window_intervals
            ]

    return cheapest_window


def find_n_cheapest_hours(prices, n=8):
    """
    Finds the N cheapest individual hours.

    Parameters:
    - prices: dict with (start, stop) tuples as keys and prices as values.
    - n: number of cheapest hours to find.

    Returns:
    - List of (start, stop, price) tuples sorted by price ascending.
    """
    sorted_prices = sorted(
        [(start, stop, price) for (start, stop), price in prices.items()],
        key=lambda x: x[2],
    )
    return sorted_prices[:n]


def categorize_prices_into_quadrants(hourly_prices):
    """
    Categorizes hourly prices into four quadrants: Cheapest, Cheap, Expensive, Most Expensive.

    Parameters:
    - hourly_prices: Dict with (start_time, stop_time) tuples as keys and prices as values.

    Returns:
    - Dict with quadrant names as keys and lists of (start_time, stop_time, price) tuples as values.
    """
    prices = list(hourly_prices.values())
    min_price = min(prices)
    max_price = max(prices)
    interval = (max_price - min_price) / 4

    quadrants = {"Cheapest": [], "Cheap": [], "Expensive": [], "Most Expensive": []}

    for (start, stop), price in hourly_prices.items():
        if price < min_price + interval:
            quadrants["Cheapest"].append((start, stop, price))
        elif price < min_price + 2 * interval:
            quadrants["Cheap"].append((start, stop, price))
        elif price < min_price + 3 * interval:
            quadrants["Expensive"].append((start, stop, price))
        else:
            quadrants["Most Expensive"].append((start, stop, price))

    return quadrants
