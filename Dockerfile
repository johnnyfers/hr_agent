# syntax=docker/dockerfile:1.7
# Multi-stage so the runtime image doesn't carry build tools.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY pyproject.toml ./
COPY src ./src

RUN pip install --prefix=/install ".[postgres]"


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/install/lib/python3.12/site-packages"

# curl for healthcheck. tini for proper signal handling under PID 1.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /install
COPY src ./src
COPY static ./static

# Non-root user.
RUN useradd --create-home --uid 1000 app \
 && chown -R app:app /app
USER app

ENV HR_AGENT_DB_PATH=/app/hr_agent.db \
    HR_AGENT_LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/health || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "hr_agent.server:app", "--host", "0.0.0.0", "--port", "8000"]
