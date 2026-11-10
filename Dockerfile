# The compiler, lawc, the benchmarks' executables and the gates on Linux, on the CPU (no vcpkg, no sandbox):
#   docker build -t field-engine . && docker run --rm field-engine
# runs the fast gates (ctest -LE slow) and bench/verify.py (convergence orders against exact solutions on the CPU
# reference; results in bench/results/verify_cpu.md). With an NVIDIA card and the container toolkit (`--gpus all`) the
# CUDA backend is found at run time and verify.py checks it too; NVRTC must be on the library path
# (FIELD_NVRTC=/path/to/libnvrtc.so.12).
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake ninja-build g++ libomp-dev python3 python3-numpy ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY CMakeLists.txt CMakePresets.json requirements.txt ./
COPY src ./src
COPY demo ./demo
COPY tools ./tools
COPY bench ./bench
COPY assets ./assets

RUN cmake --preset linux-core && cmake --build --preset linux-core

CMD ["sh", "-c", "ctest --preset linux-core -LE slow --timeout 900 && cd build && python3 ../bench/verify.py"]
