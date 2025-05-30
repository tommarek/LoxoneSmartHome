FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app

# Set the working directory to where main.py is located
WORKDIR /app

CMD ["python", "main.py"]