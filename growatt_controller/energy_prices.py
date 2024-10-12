import requests
from datetime import datetime, timedelta
import logging


# Fetch energy prices data from OTE
def fetch_energy_prices():
    # Calculate the next day's date
    next_day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Construct the URL with the next day's date
    url = f"https://www.ote-cr.cz/en/short-term-markets/electricity/day-ahead-market/@@chart-data?report_date={next_day}"
    logging.info(f"Fetching energy prices from: {url}")

    # Fetch the data
    response = requests.get(url)
    data = response.json()["data"]["dataLine"][1]["point"]

    # Extract hourly prices
    hourly_prices = {int(point["x"]) - 1: float(point["y"]) for point in data}
    return hourly_prices


def find_cheapest_3_consecutive_hours(hourly_prices):
    """Finds the 3 consecutive hours with the lowest total price."""
    if len(hourly_prices) < 3:
        return []  # Not enough data to find 3 consecutive hours

    cheapest_window = []
    min_price_sum = float("inf")  # Set to infinity initially

    # Loop through hourly prices to find the 3 consecutive hours with the lowest sum
    for i in range(len(hourly_prices) - 2):
        # Sum of 3 consecutive hours
        price_sum = hourly_prices[i] + hourly_prices[i + 1] + hourly_prices[i + 2]
        if price_sum < min_price_sum:
            min_price_sum = price_sum
            cheapest_window = [i, i + 1, i + 2]

    return cheapest_window
