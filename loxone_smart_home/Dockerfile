FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

# Create the package structure that the imports expect
RUN mkdir -p /app/loxone_smart_home
COPY . /app/loxone_smart_home/

ENV PYTHONPATH=/app

CMD ["python", "-m", "loxone_smart_home.main"]