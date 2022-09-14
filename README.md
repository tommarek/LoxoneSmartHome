# LoxoneSmartHome

Aim of this repo is to grab data from Loxone smart home and a PHV array with Growatt battery and inverter and use the data to predict excess of solar energy.
This information will then be sent back to loxone so it can trigger heating/cooling based on that.

## Docker containers

### Grafana

Grafana service connected to influxdb database that displays relevant data

### InfluxDB

Database used by all the services to store data

### Loxone2DB

Rust script that reads data pushed by Loxone server and stores them to the DB

### Mosquitto

MQTT broker used to collect data from Growatt (https://github.com/otti/Growatt_ShineWiFi-S) and possibly other sensors

### WeatherForecastScraper (TODO)

get the weather forecast data and store them to InfluxDB

### Mastermind (TODO)

Python application that will read data from the InfluxDB and sends the heating/cooling commands to the Loxone server.
