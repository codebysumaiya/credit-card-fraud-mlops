
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers output immediately
# (so logs show up right away in `docker logs`, not delayed)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (separate layer) so Docker can cache this step
# and skip re-installing every package every time you only change app code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual project files
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY params.yaml .

EXPOSE 8000

# Basic healthcheck so `docker ps` and orchestrators know if the app died
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]