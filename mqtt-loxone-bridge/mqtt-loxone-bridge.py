import paho.mqtt.client as mqtt
import socket
import json
import os

# MQTT broker details
mqtt_host = os.getenv("MQTT_HOST", 'mosquitto')
mqtt_port = int(os.getenv("MQTT_PORT", "1883"))
mqtt_topic = os.getenv("MQTT_TOPIC", "energy/solar")

# Loxone server details
loxone_host = os.getenv("LOXONE_HOST", "192.168.0.200")
loxone_port = int(os.getenv("LOXONE_PORT", "4000"))

# Define the MQTT client
mqtt_client = mqtt.Client()

# Define the on_connect callback function for the MQTT client
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(mqtt_topic)

# Define the on_message callback function for the MQTT client
def on_message(client, userdata, msg):
    #print(f"Received message on topic {msg.topic}: {msg.payload}")
    if msg.topic == mqtt_topic:
        data = json.loads(msg.payload)
        message = ';'.join([f"{k}={v}" for k, v in data.items()])
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.sendto(bytes(message, "utf-8"), (loxone_host, loxone_port))
        print(f"Message sent")

# Set the MQTT client callbacks
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# Connect to the MQTT broker
mqtt_client.connect(mqtt_host, mqtt_port, 30)

# Start the MQTT client loop
mqtt_client.loop_forever()
