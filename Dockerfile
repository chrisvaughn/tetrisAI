FROM python:3.14-slim
WORKDIR /usr/app

RUN apt update && apt install -y \
  libgl1  \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /usr/local/bin/uv

COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
