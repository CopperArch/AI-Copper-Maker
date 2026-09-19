FROM python:3.14-slim

WORKDIR /app/backend

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libnotify-bin \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

EXPOSE 8081

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081", "--timeout-graceful-shutdown", "10"]