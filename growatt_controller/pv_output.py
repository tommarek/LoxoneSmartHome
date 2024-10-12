import math
import time
import requests
from datetime import datetime

# Constants
LATITUDE = 49.4949875
LONGITUDE = 17.4301853
PANEL_AREA = 2.56  # in m²
PANEL_EFFICIENCY = 0.213
NUM_PANELS_SE = 11
NUM_PANELS_SW = 14
PANEL_TILT = 35  # in degrees
AZIMUTH_SE = 143  # in degrees
AZIMUTH_SW = 234  # in degrees
PI = math.pi


# Function to calculate the day of the year from Unix timestamp
def calculate_day_of_year(timestamp):
    dt = datetime.utcfromtimestamp(timestamp)
    return dt.timetuple().tm_yday


# Function to calculate the hourly zenith angle
def calculate_zenith_angle(latitude, longitude, day_of_year, hour):
    declination = 23.44 * math.sin((360.0 / 365.0) * (day_of_year - 81) * PI / 180)
    solar_time = hour + (4 * (longitude - 15.0 * (int(longitude / 15.0))) / 60.0)
    hour_angle = (solar_time - 12) * 15
    zenith_angle = (
        math.acos(
            math.sin(latitude * PI / 180) * math.sin(declination * PI / 180)
            + math.cos(latitude * PI / 180)
            * math.cos(declination * PI / 180)
            * math.cos(hour_angle * PI / 180)
        )
        * 180
        / PI
    )
    return zenith_angle


# Function to calculate the daily power output
def calculate_daily_power(total_irradiance, day_of_year):
    total_power = 0
    valid_hours = 0

    for hour in range(24):
        zenith_angle = calculate_zenith_angle(LATITUDE, LONGITUDE, day_of_year, hour)
        if zenith_angle < 90:  # Only consider hours when the sun is above the horizon
            gamma_se = AZIMUTH_SE - 180
            gamma_sw = AZIMUTH_SW - 180
            cos_theta_se = math.sin((90 - zenith_angle) * PI / 180) * math.sin(
                PANEL_TILT * PI / 180
            ) * math.cos(gamma_se * PI / 180) + math.cos(
                (90 - zenith_angle) * PI / 180
            ) * math.cos(PANEL_TILT * PI / 180)
            cos_theta_sw = math.sin((90 - zenith_angle) * PI / 180) * math.sin(
                PANEL_TILT * PI / 180
            ) * math.cos(gamma_sw * PI / 180) + math.cos(
                (90 - zenith_angle) * PI / 180
            ) * math.cos(PANEL_TILT * PI / 180)
            power_se = cos_theta_se * PANEL_AREA * PANEL_EFFICIENCY
            power_sw = cos_theta_sw * PANEL_AREA * PANEL_EFFICIENCY
            total_power += (power_se * NUM_PANELS_SE) + (power_sw * NUM_PANELS_SW)
            valid_hours += 1

    # Scale total power by the total irradiance divided by the number of valid hours
    if valid_hours > 0:
        total_power = total_power * total_irradiance / valid_hours

    return total_power


# Fetch irradiance data from PVForecast.cz
def fetch_irradiance_data():
    url = "https://www.pvforecast.cz/api/?key=a4qtew&lat=49.495&lon=17.431&format=simple&type=day&number=3"
    response = requests.get(url)
    data = response.text.split("|")
    irradiance_today = float(data[1])
    irradiance_tomorrow = float(data[2])
    irradiance_day_after = float(data[3])
    return irradiance_today, irradiance_tomorrow, irradiance_day_after


# Function to calculate power output for the next three days
def get_forecasted_power():
    irradiance_today, irradiance_tomorrow, irradiance_day_after = (
        fetch_irradiance_data()
    )

    unix_timestamp = int(time.time())
    day_of_year_today = calculate_day_of_year(unix_timestamp)
    day_of_year_tomorrow = (day_of_year_today + 1) % 365
    day_of_year_day_after = (day_of_year_today + 2) % 365

    total_power_today = calculate_daily_power(irradiance_today, day_of_year_today)
    total_power_tomorrow = calculate_daily_power(
        irradiance_tomorrow, day_of_year_tomorrow
    )
    total_power_day_after = calculate_daily_power(
        irradiance_day_after, day_of_year_day_after
    )

    return total_power_today, total_power_tomorrow, total_power_day_after
