FROM python:3.9-slim
WORKDIR /usr/app

RUN apt update && apt install -y libgl1 && rm -rf /var/lib/apt/lists/*

RUN pip3 install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml .
RUN poetry install --no-dev

COPY . .
