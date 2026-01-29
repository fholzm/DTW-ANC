ARG UV_VERSION=latest


FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv


FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-24.04

ENV PYTHONDONTWRITEBYTECODE=True \
    PYTHONUNBUFFERED=True \
    UV_LINK_MODE=copy

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    bzip2 \
    coreutils \
    curl \
    ca-certificates \
    git \
    git-lfs \
    jackd \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    htop \
    mercurial \
    openssh-client \
    procps \
    puredata \
    software-properties-common \
    subversion \
    wget \
    zsh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- http://apt.hoertech.de/openmha-packaging.pub | tee /etc/apt/trusted.gpg.d/openmha-packaging.asc
RUN apt-add-repository 'deb [arch=amd64] http://apt.hoertech.de noble universe'

RUN apt-get update -q
RUN apt-get install -y -q tascarpro

COPY .devcontainer/start_jack.sh /usr/local/bin/start_jack.sh
RUN chmod +x /usr/local/bin/start_jack.sh

COPY --from=uv --chown=vscode: /uv /uvx /bin/
