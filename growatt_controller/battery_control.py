import json
import logging
import paho.mqtt.client as mqtt

# MQTT broker details
BROKER_ADDRESS = "your_broker_address"
MQTT_PORT = 1883

# MQTT topics
BATTERY_FIRST_TOPIC = "energy/solar/command/batteryfirst/set/timeslot"
AC_CHARGE_TOPIC = "energy/solar/command/batteryfirst/set/acchargeenabled"
DISABLE_BATTERY_FIRST_TOPIC = "energy/solar/command/batteryfirst/set/timeslot"

# Global simulation flag
SIMULATE = False  # Default: not in simulate mode

# Initialize MQTT client
client = mqtt.Client()


def set_simulate_mode(simulate):
    """
    Set the simulation mode globally.
    """
    global SIMULATE
    SIMULATE = simulate


def set_battery_first(start_hour, stop_hour):
    """
    Set battery-first mode for a specified time window.
    - start_hour and stop_hour should be in the format "HH:MM"
    """
    if SIMULATE:
        logging.info(
            f"[SIMULATE] Would set battery-first mode from {start_hour} to {stop_hour}"
        )
        return

    # Define the payload for the battery-first time window
    payload = {"start": start_hour, "stop": stop_hour, "enabled": True, "slot": 1}

    # Convert to JSON and send via MQTT
    client.publish(BATTERY_FIRST_TOPIC, json.dumps(payload))
    logging.info(f"Battery-first mode scheduled from {start_hour} to {stop_hour}")


def enable_ac_charge():
    """
    Enable AC charging during the battery-first mode.
    """
    if SIMULATE:
        logging.info("[SIMULATE] Would enable AC charging.")
        return

    # Define the payload to enable AC charge
    payload = {"value": True}

    # Convert to JSON and send via MQTT
    client.publish(AC_CHARGE_TOPIC, json.dumps(payload))
    logging.info("AC charging enabled")


def disable_battery_first():
    """
    Disable battery-first mode by setting enabled to False for the time slot.
    """
    if SIMULATE:
        logging.info("[SIMULATE] Would disable battery-first mode.")
        return

    # Define the payload to disable battery-first mode
    payload = {
        "start": "00:00",  # Time doesn't matter as we're disabling the slot
        "stop": "00:00",
        "enabled": False,
        "slot": 1,
    }

    # Convert to JSON and send via MQTT
    client.publish(DISABLE_BATTERY_FIRST_TOPIC, json.dumps(payload))
    logging.info("Battery-first mode disabled")


def configure_battery_first_with_ac_charge(start_hour, stop_hour):
    """
    Combine setting battery-first mode with enabling AC charge.
    """
    set_battery_first(start_hour, stop_hour)
    enable_ac_charge()


# Connect to the MQTT broker
def connect_mqtt():
    if SIMULATE:
        logging.info("[SIMULATE] Would connect to MQTT broker.")
        return
    client.connect(BROKER_ADDRESS, MQTT_PORT, 60)
    client.loop_start()


# Disconnect from the MQTT broker
def disconnect_mqtt():
    if SIMULATE:
        logging.info("[SIMULATE] Would disconnect from MQTT broker.")
        return
    client.loop_stop()
    client.disconnect()
