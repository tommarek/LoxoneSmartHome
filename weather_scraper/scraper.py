#!/usr/bin/env python3

import os
import time
import requests
import json
from collections import namedtuple
from influxdb_client import InfluxDBClient, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

import paho.mqtt.client as mqtt


USE_SERVICE = os.getenv('USE_SERVICE', 'openweathermap')

USE_MQTT = os.getenv('USE_MQTT', True)
MQTT_HOST = os.getenv('MQTT_HOST', '192.168.0.201')
MQTT_PORT = os.getenv('MQTT_PORT', 1883)
MQTT_TOPIC = os.getenv('MQTT_TOPIC', 'weather')

INFLUXDB_HOST = os.getenv('INFLUXDB_HOST', '192.168.0.201:8086')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', 'loxone')
INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET', 'loxone')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', '7HrEuj8kzOS1f-0mjU_GT4hS_9gHfdjUT6j5QAM22oDg0z44DsxmiveTGMqTa0Zl1QezDh132utLbXi-IL8h9A==')

POLLING_INTERVAL = os.getenv('POLLING_INTERVAL', 3600)
SIMULATE = os.getenv('SIMULATE', False)
LATITUDE = os.getenv('LATITUDE', 49.4949522)
LONGITUDE = os.getenv('LONGITUDE', 17.4302361)

ALADIN_URL = 'https://aladinonline.androworks.org/get_data.php?latitude={latitude}&longitude={longitude}'
OPENWEATHERMAP_URL = 'https://api.openweathermap.org/data/3.0/onecall?lat={latitude}&lon={longitude}&exclude=minutely,daily,alerts&units=metric&appid={api_key}'

OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', None)


HourlyData = namedtuple('HourlyData', [
    'temp',
    'precip',
    'wind',
    'wind_direction',
    'clouds',
    'rh',
    'pressure',
])


def get_aladin_data():
    resp = requests.get(ALADIN_URL.format(latitude=LATITUDE, longitude=LONGITUDE))
    js = resp.json()
    return {
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
    return {
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

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

def publish_MQTT(client, topic, payload):
    client.publish(topic, payload)

def save_to_inlfuxdb(write_api, data):
    write_api.write(
        INFLUXDB_BUCKET,
        INFLUXDB_ORG,
        [{
            "measurement": "weather_forecast",
            "tags": {
                "room": "outside"
            },
            "fields": {name: float(value) for name, value in data['hourly'][0].items()},
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
    }[USE_SERVICE]

    print("Starting weather scraper")
    while True:
        data = get_data_fn()

        if USE_MQTT:
            if mqtt_client.is_connected():
                print("Publishing to MQTT topic {}".format(MQTT_TOPIC))
                mqtt_client.publish(MQTT_TOPIC, json.dumps(data))
            else:
                print("Reconnecting to MQTT broker")
                mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)

        # save the weather forecast to InfluxDB
        save_to_inlfuxdb(influx_write_api, data)

        time.sleep(int(POLLING_INTERVAL))
