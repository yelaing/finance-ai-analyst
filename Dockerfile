FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENV=prod

WORKDIR /app

# 依赖单独一层：只要 requirements.txt 不变，改代码不会触发重装
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend

# 非 root 运行；/app/data 预建并授权，配合 compose 里的命名卷使用
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data/chroma \
    && chown -R app:app /app
USER app

EXPOSE 8000 8501

# 存活探针：只验证进程能响应，依赖状态看 /health/ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "backend.main"]
