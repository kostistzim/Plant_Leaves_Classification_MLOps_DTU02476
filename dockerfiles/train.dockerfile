# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src/ src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY configs/ configs/


WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_dev.txt
RUN pip install . --no-deps --no-cache-dir --verbose

# ENTRYPOINT ["python", "-u"]

# Default to train.py
#CMD ["src/plant_leaves/train.py"]
ENTRYPOINT ["sh", "-c", "python -u $(pwd)/src/plant_leaves/train.py"]
