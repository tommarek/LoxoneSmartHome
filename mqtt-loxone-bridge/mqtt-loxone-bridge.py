import paho.mqtt.client as mqtt
import socket
import json
import os

# MQTT broker details
mqtt_host = os.getenv("MQTT_HOST", 'mosquitto')
mqtt_port = int(os.getenv("MQTT_PORT", "1883"))
mqtt_topics = os.getenv("MQTT_TOPICS", "energy/solar,teplomer/TC").split(',') # comma separated list of topics

# Loxone server details
loxone_host = os.getenv("LOXONE_HOST", "192.168.0.200")
loxone_port = int(os.getenv("LOXONE_PORT", "4000"))

# Define the MQTT client
mqtt_client = mqtt.Client()

# Define the on_connect callback function for the MQTT client
def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    # Subscribing to multiple topics
    for topic in mqtt_topics:
        client.subscribe(topic)

# Define the on_message callback function for the MQTT client
def on_message(client, userdata, msg):
    if msg.topic in mqtt_topics:
        try:
            data = json.loads(msg.payload)
            if isinstance(data, dict):
                message = ';'.join([f"{k}={v}" for k, v in data.items()])
            else:
                message = f"{msg.topic}={data}"
            udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp.sendto(bytes(message, "utf-8"), (loxone_host, loxone_port))
            #print(f"Message {message} sent")
        except Exception as e:
            print("Error sending message:", e)

# Set the MQTT client callbacks
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# Connect to the MQTT broker
mqtt_client.connect(mqtt_host, mqtt_port, 30)

# Start the MQTT client loop
mqtt_client.loop_forever()
