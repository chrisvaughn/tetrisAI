FROM python:3.9
WORKDIR /usr/app

RUN apt update && apt install -y libgl1
RUN pip3 install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml .
RUN poetry install --no-dev

COPY . .
