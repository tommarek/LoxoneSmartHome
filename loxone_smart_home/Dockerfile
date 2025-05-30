FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app/loxone_smart_home

ENV PYTHONPATH=/app

CMD ["python", "-m", "loxone_smart_home.main"]