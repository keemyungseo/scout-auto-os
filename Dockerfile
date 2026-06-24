FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/research_bundle \
    SCOUT_DATA_DIR=/app/data \
    SCOUT_LOG_DIR=/app/logs \
    SCOUT_UNIVERSE_CACHE_ONLY=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./scout_auto_os/
COPY entrypoint.py .
COPY research_bundle/ ./research_bundle/

RUN mkdir -p /app/data /app/logs

CMD ["python", "entrypoint.py"]
