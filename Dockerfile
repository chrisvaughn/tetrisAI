FROM python:3.11-slim
WORKDIR /usr/app

RUN apt update && apt install -y \
  libgl1  \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install poetry
RUN poetry config virtualenvs.create false

COPY poetry.lock pyproject.toml .
RUN poetry install --no-dev

COPY . .
