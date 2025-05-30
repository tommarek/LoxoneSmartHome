FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

# Copy the current directory contents (which is already the loxone_smart_home package)
COPY . /app/

# Add a debug step to see what we have
RUN ls -la /app/

ENV PYTHONPATH=/app

CMD ["python", "main.py"]