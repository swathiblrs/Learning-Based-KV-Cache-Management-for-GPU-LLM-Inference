FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 USE_TF=0 HF_HOME=/tmp/huggingface
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[serving]' && useradd --uid 10001 --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz')"
CMD ["uvicorn", "kv_cache.serving:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
