FROM python:3.9

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY mqtt-loxone-bridge.py /app

CMD ["python", "mqtt-loxone-bridge.py"]
