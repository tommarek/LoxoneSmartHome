FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app

# Use the integrated launcher that runs both main app and web service
CMD ["python", "run_integrated.py"]