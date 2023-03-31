import paho.mqtt.client as mqtt
import requests
import json
import os

# MQTT broker details
mqtt_host = os.getenv("MQTT_HOST", 'mosquitto')
mqtt_port = int(os.getenv("MQTT_PORT", "1883"))
mqtt_topic = os.getenv("MQTT_TOPIC", "energy/solar")

# Loxone server details
loxone_host = os.getenv("LOXONE_HOST", "192.168.0.200")
loxone_port = int(os.getenv("LOXONE_PORT", "4000"))
loxone_username = os.getenv("LOXONE_USERNAME", "")
loxone_password = os.getenv("LOXONE_PASSWORD", "")

# Loxone API endpoint
loxone_url = f"http://{loxone_host}:{loxone_port}/dev/sps/io/{loxone_username}/{loxone_password}"

# Define the MQTT client
mqtt_client = mqtt.Client()

# Define the on_connect callback function for the MQTT client
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(mqtt_topic)

# Define the on_message callback function for the MQTT client
def on_message(client, userdata, msg):
    print(f"Received message on topic {msg.topic}: {msg.payload}")
    if msg.topic == mqtt_topic:
        data = json.loads(msg.payload)
        for k, v in data.items():
            response = requests.post(f"{mqtt_topic}/{k}={v}")
            if response.status_code == 200:
                print(f"Message sent to Loxone server for topic {mqtt_topic}/{k}")
            else:
                print(f"Error sending message to Loxone server for topic {mqtt_topic}/{k}: {response.status_code} {response.text}")

# Set the MQTT client callbacks
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# Connect to the MQTT broker
mqtt_client.connect(mqtt_host, mqtt_port, 30)

# Start the MQTT client loop
mqtt_client.loop_forever()