#!/usr/bin/env python3

import os
import time
import datetime
import requests
import json
from collections import namedtuple
from influxdb_client import InfluxDBClient, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

import paho.mqtt.client as mqtt


USE_SERVICE = os.getenv('USE_SERVICE', 'openmeteo')

USE_MQTT = os.getenv('USE_MQTT', True)
MQTT_HOST = os.getenv('MQTT_HOST', '192.168.0.201')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC = os.getenv('MQTT_TOPIC', 'weather')

INFLUXDB_HOST = os.getenv('INFLUXDB_HOST', '192.168.0.201:8086')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', 'loxone')
INFLUXDB_BUCKET = os.getenv('INFLUXDB_WEATHER_BUCKET', 'weather_forecast')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', '7HrEuj8kzOS1f-0mjU_GT4hS_9gHfdjUT6j5QAM22oDg0z44DsxmiveTGMqTa0Zl1QezDh132utLbXi-IL8h9A==')

POLLING_INTERVAL = os.getenv('POLLING_INTERVAL', 3600)
SIMULATE = os.getenv('SIMULATE', False)
LATITUDE = os.getenv('LATITUDE', 49.4949522)
LONGITUDE = os.getenv('LONGITUDE', 17.4302361)

ALADIN_URL = 'https://aladinonline.androworks.org/get_data.php?latitude={latitude}&longitude={longitude}'

OPENWEATHERMAP_URL = 'https://api.openweathermap.org/data/3.0/onecall?lat={latitude}&lon={longitude}&exclude=minutely,daily,alerts&units=metric&appid={api_key}'
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', None)

OPEN_METEO_URL = 'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly={fields}&windspeed_unit=ms&timeformat=unixtime&timezone=GMT'
OPEN_METEO_FIELDS = 'temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,showers,snowfall,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,visibility,windspeed_10m,winddirection_10m,windgusts_10m,temperature_80m,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,terrestrial_radiation,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant&models=best_match&daily=sunrise,sunset,precipitation_sum,rain_sum,precipitation_hours,shortwave_radiation_sum'

HourlyData = namedtuple('HourlyData', [
    'temp',
    'precip',
    'wind',
    'wind_direction',
    'clouds',
    'rh',
    'pressure',
])


def get_openmeteo_data():
    resp = requests.get(OPEN_METEO_URL.format(latitude=LATITUDE, longitude=LONGITUDE, fields=OPEN_METEO_FIELDS))
    js = resp.json()
    OpenMeteoHourlyRecord = namedtuple('OpenMeteoHourlyRecord', list(js['hourly'].keys()))
    OpenMeteoDailyRecord = namedtuple('OpenMeteoDailyRecord', list(js['daily'].keys()))
    hour_now = int(datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0).timestamp())
    hourly_data = {t[0]: OpenMeteoHourlyRecord(*t) for t in zip(*(js['hourly'][k] for k in js['hourly'].keys()))}
    day_now = int(datetime.datetime.now(datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    daily_data = {t[0]: OpenMeteoDailyRecord(*t) for t in zip(*(js['daily'][k] for k in js['daily'].keys()))}

    now = int(datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).timestamp())
    return {
        'source': 'openmeteo',
        'timestamp': now,
        'now': hourly_data[hour_now]._asdict(),
        'today': daily_data[day_now]._asdict(),
        'hourly': [i._asdict() for i in hourly_data.values() if i.time >= hour_now],
        'daily': [i._asdict() for i in daily_data.values()],
    }

def get_aladin_data():
    resp = requests.get(ALADIN_URL.format(latitude=LATITUDE, longitude=LONGITUDE))
    js = resp.json()
    return {
        'source': 'aladin',
        'timestamps': js['nowCasting']['nowUtc'],
        'hourly': [
            HourlyData(*data)._asdict() for data in zip(
                js['parameterValues']['TEMPERATURE'],
                js['parameterValues']['PRECIPITATION_TOTAL'],
                js['parameterValues']['WIND_SPEED'],
                js['parameterValues']['WIND_DIRECTION'],
                js['parameterValues']['CLOUDS_TOTAL'],
                js['parameterValues']['HUMIDITY'],
                js['parameterValues']['PRESSURE'],
            )
        ]
    }

def get_openweathermap_data():
    resp = requests.get(OPENWEATHERMAP_URL.format(latitude=LATITUDE, longitude=LONGITUDE, api_key=OPENWEATHERMAP_API_KEY))
    js = resp.json()
    data = {
        'source': 'openweathermap',
        'timestamp': js['hourly'][0]['dt'],
        'hourly': [
            HourlyData(
                data['temp'],
                data.get('rain', {}).get('1h', 0),
                data['wind_speed'],
                data['wind_deg'],
                data['clouds'],
                data['humidity'],
                data['pressure']
            )._asdict() for data in js['hourly']]
    }
    # update first hour forecast to current weather
    data['hourly'][0] = HourlyData(
        js['current']['temp'],
        js['current'].get('rain', {}).get('1h', 0),
        js['current']['wind_speed'],
        js['current']['wind_deg'],
        js['current']['clouds'],
        js['current']['humidity'],
        js['current']['pressure']
    )._asdict()
    return data

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

def publish_MQTT(client, topic, payload):
    client.publish(topic, payload)

def save_to_inlfuxdb(write_api, data, forecast_type):
    write_api.write(
        INFLUXDB_BUCKET,
        INFLUXDB_ORG,
        [{
            "measurement": "weather_forecast",
            "tags": {
                "room": "outside",
                "type": forecast_type
            },
            "fields": {name: float(value) for name, value in data['data'].items()},
            "time": data['timestamp']
        }],
        WritePrecision.S
    )

if __name__ == '__main__':
    # setup MQTT
    mqtt_client = mqtt.Client("weather_scraper")
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    if USE_MQTT:
        mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
        mqtt_client.loop_start()

    # setup InfluxDB
    influx_client = InfluxDBClient(url=INFLUXDB_HOST, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG, debug=False)
    influx_write_api = influx_client.write_api(write_options=SYNCHRONOUS)

    get_data_fn = {
        'aladin': get_aladin_data,
        'openweathermap': get_openweathermap_data,
        'openmeteo': get_openmeteo_data,
    }[USE_SERVICE]

    print("Starting weather scraper")
    while True:
        raw_data = get_data_fn()
        mqtt_data = {}
        db_data_now = {}
        db_data_today = {}

        if raw_data['source'] == 'openmeteo':
            db_data_now = {
                'timestamp': raw_data['timestamp'],
                'source': raw_data['source'],
                'data': raw_data['now'],
            }
            db_data_today = {
                'timestamp': raw_data['timestamp'],
                'source': raw_data['source'],
                'data': raw_data['today'],
            }
            mqtt_data = {
                'timestamp': raw_data['timestamp'],
                'source': raw_data['source'],
                'hourly': raw_data['hourly'],
                'daily': raw_data['daily'],
            }
        else:
            db_data_now = {
                'timestamp': raw_data['timestamp'],
                'source': raw_data['source'],
                'data': raw_data['now'],
            }
            mqtt_data = {
                'timestamp': raw_data['timestamp'],
                'source': raw_data['source'],
                'hourly': raw_data['hourly'],
            }

        if USE_MQTT:
            if mqtt_client.is_connected():
                print("Publishing to MQTT topic {}".format(MQTT_TOPIC))
                mqtt_client.publish(MQTT_TOPIC, json.dumps(mqtt_data))
            else:
                print("Reconnecting to MQTT broker")
                mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)

        # save the weather forecast to InfluxDB
        print('Saving data to influxDB timestamp: {}'.format(db_data_now['timestamp']))
        save_to_inlfuxdb(influx_write_api, db_data_now, 'hour')
        save_to_inlfuxdb(influx_write_api, db_data_today, 'day')

        time.sleep(int(POLLING_INTERVAL))
