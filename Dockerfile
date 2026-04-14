# 1. Base Image: Use Ubuntu 22.04 which supports GLIBC 2.34 and libffi.so.8
FROM ubuntu:22.04

# Disable interactive prompts (e.g., timezone setup)
ENV DEBIAN_FRONTEND=noninteractive

# 2. Install essential libraries and tools
# Explicitly install libffi8 and libreadline8
RUN apt-get update && apt-get install -y \
    software-properties-common \
    libtcl8.6 \
    libreadline8 \
    libffi8 \
    tcl-dev \
    zlib1g \
    make gcc g++ bc \
    wget git vim \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Python packages required for AE paper plotting and data processing
RUN pip3 install matplotlib numpy pandas

# 4. Set environment variables and working directory
ENV SMOOTH_HOME=/workspace/SMOOTH
WORKDIR /workspace/SMOOTH

# 5. Default command when the container starts
CMD ["/bin/bash"]

