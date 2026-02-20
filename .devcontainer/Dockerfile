ARG UV_VERSION=latest


FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv


FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

ENV PYTHONDONTWRITEBYTECODE=True \
    PYTHONUNBUFFERED=True \
    UV_LINK_MODE=copy

# hadolint ignore=DL3008
RUN apt-get update \
    && echo "jackd jackd/tweak_rt_limits boolean true" | debconf-set-selections \
    && echo "msttcorefonts msttcorefonts/accepted-mscorefonts-eula boolean true" | debconf-set-selections \
    && apt-get install -y --no-install-recommends\
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
    htop \
    mercurial \
    msttcorefonts \
    openssh-client \
    pkg-config \
    procps \
    puredata \
    software-properties-common \
    subversion \
    wget \
    zsh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo "GROUP ( libbsd.so )" > /usr/lib/x86_64-linux-gnu/libbsd_auto.so \
    && echo "/usr/lib/x86_64-linux-gnu/libbsd_auto.so" \
       > /etc/ld.so.conf.d/libbsd_auto.conf \
    && ldconfig \
    && ln -s /usr/lib/x86_64-linux-gnu/libbsd.so /usr/lib/x86_64-linux-gnu/libbsd.so.1

RUN wget -qO- http://apt.hoertech.de/openmha-packaging.pub | sudo tee /etc/apt/trusted.gpg.d/openmha-packaging.asc
RUN sudo apt-add-repository -y 'deb [arch=amd64] http://apt.hoertech.de jammy universe'

RUN apt-get update -q
RUN apt-get install -y -q tascarpro

COPY .devcontainer/start_jack.sh /usr/local/bin/start_jack.sh
RUN chmod +x /usr/local/bin/start_jack.sh

COPY --from=uv --chown=vscode: /uv /uvx /bin/
