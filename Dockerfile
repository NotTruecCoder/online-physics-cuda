# The engine, lawc, the benchmarks' executables and the gates on Linux, on the CPU (no vcpkg, no sandbox):
#   docker build -t field-engine . && docker run --rm field-engine
# With an NVIDIA card and the container toolkit (`--gpus all`) the CUDA backend is found at run time; NVRTC must be on
# the library path (FIELD_NVRTC=/path/to/libnvrtc.so.12).
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake ninja-build g++ libomp-dev python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY CMakeLists.txt CMakePresets.json requirements.txt ./
COPY src ./src
COPY demo ./demo
COPY tools ./tools
COPY bench ./bench
COPY assets ./assets

RUN cmake --preset linux-core && cmake --build --preset linux-core

CMD ["ctest", "--preset", "linux-core"]
