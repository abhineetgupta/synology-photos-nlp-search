ARG BASE_CONTAINER=python:3.10-slim
FROM $BASE_CONTAINER

# Install required packages
RUN pip3 install -U pip \
    && \
    pip3 install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
    && \
    pip3 install --no-cache-dir \
    black \
    isort \
    sentence_transformers

# Copy files
COPY syno_smart_search.py /scripts/syno_smart_search.py
COPY utils.py /scripts/utils.py
COPY syno_generate_embeddings.py /scripts/syno_generate_embeddings.py
COPY synoapi.py /scripts/synoapi.py

WORKDIR "/scripts"
