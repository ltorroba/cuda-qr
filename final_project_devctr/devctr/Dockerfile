FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
LABEL maintainer="Nikita Lazarev <niklz@csail.mit.edu"

ARG DEBIAN_FRONTEND=noninteractive

# Install needed packages.
RUN apt update && \
    apt -y install python3 && \
    apt -y install git

# Get CUTLASS:
#   note: it will be placed in the root of the container, e.g. /cutlass
RUN git clone https://github.com/NVIDIA/cutlass.git
