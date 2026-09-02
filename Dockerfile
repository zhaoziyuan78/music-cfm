FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
WORKDIR /workspace
COPY pyproject.toml uv.lock* README.md ./
COPY configs ./configs
COPY src ./src
RUN uv sync --no-dev
ENTRYPOINT ["uv", "run"]
