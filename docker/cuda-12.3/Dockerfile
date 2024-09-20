FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    # python build dependencies \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python

ARG PYTHON_VERSION=3.12.5

RUN curl https://pyenv.run | bash

ENV HOME=/root
ENV PATH=${HOME}/.pyenv/shims:${HOME}/.pyenv/bin:${PATH}

RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION} && \
    pyenv rehash && \
    pip install --no-cache-dir -U pip setuptools wheel && \
    pip install packaging ninja

# Install PyTorch
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# Install other Python libraries
RUN pip install -U transformers yt-dlp setuptools sphn patchelf
RUN pip install git+https://github.com/mobiusml/hqq.git
