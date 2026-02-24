# CodeOcean Environment Dockerfile
# Adapted from .devcontainer/Dockerfile

FROM registry.codeocean.com/codeocean/ubuntu-desktop:ubuntu-22.04-guacamole-1.5.2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=True \
    PYTHONUNBUFFERED=True \
    UV_LINK_MODE=copy

# System packages (audio tooling, build deps, FAUST, JACK, TASCAR)
RUN apt-get update \
    && echo "jackd jackd/tweak_rt_limits boolean true" | debconf-set-selections \
    && apt-get install -y --no-install-recommends \
    build-essential \
    bzip2 \
    coreutils \
    curl \
    ca-certificates \
    faust \
    git \
    git-lfs \
    jackd \
    libasound2-dev \
    libbsd-dev \
    libjack-jackd2-dev \
    libgtk2.0-dev \
    libglib2.0-0 \
    libncurses-dev \
    libncursesw6 \
    libmicrohttpd-dev \
    libsndfile-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    llvm \
    pkg-config \
    procps \
    puredata \
    software-properties-common \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# libbsd workaround (required by TASCAR)
RUN echo "GROUP ( libbsd.so )" > /usr/lib/x86_64-linux-gnu/libbsd_auto.so \
    && echo "/usr/lib/x86_64-linux-gnu/libbsd_auto.so" \
       > /etc/ld.so.conf.d/libbsd_auto.conf \
    && ldconfig \
    && ln -s /usr/lib/x86_64-linux-gnu/libbsd.so /usr/lib/x86_64-linux-gnu/libbsd.so.1

# Install TASCAR from HörTech repository
RUN wget -qO- http://apt.hoertech.de/openmha-packaging.pub | tee /etc/apt/trusted.gpg.d/openmha-packaging.asc \
    && apt-add-repository -y 'deb [arch=amd64] http://apt.hoertech.de jammy universe' \
    && apt-get update -q \
    && apt-get install -y -q tascarpro \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Python 3.12 via uv
RUN uv python install 3.12
