#!/bin/bash
set -e

echo "🚀 Deploying AI Server (Python/FastAPI)..."

# Build
docker build -t abide-ai:latest .

# Stop existing
docker stop abide-ai 2>/dev/null || true
docker rm abide-ai 2>/dev/null || true

# Run
docker run -d \
  --name abide-ai \
  --network abide-net \
  --env-file .env \
  -p 8000:8000 \
  --restart unless-stopped \
  abide-ai:latest

echo "✅ AI Server deployed on port 8000"
