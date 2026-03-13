FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Python deps (캐시 레이어)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# App code
COPY main.py .
COPY app/ app/

# .env는 docker-compose 환경변수로 주입 (로컬 개발용 폴백)
COPY .env* ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
