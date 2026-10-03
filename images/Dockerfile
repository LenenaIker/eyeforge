# The two images of the Nebius worker (docs/imagenes.md), built and pushed by images/build.sh:
#   light  for runs without synthetic data: TensorFlow (student) and torch (teachers), nothing else
#   isaac  NVIDIA's Isaac Sim image plus the same two Pythons, for runs that render
# Both get /opt/s2c from the same "envs" stage, at the same path, so the layer is identical and the worker
# script skips installing anything when it finds it. No weights inside: the teachers' come from the cache
# bucket (sim2chip.teacher.weights), the shader caches too (sim2chip.cloud_worker). The pip lists are written
# by build.sh from sim2chip/backends/nebius.py (WORKER_DEPS, TEACHER_DEPS), the one place they live.
ARG ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0

FROM ubuntu:22.04 AS envs
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/s2c/pythons UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY worker.txt teacher.txt /tmp/
RUN uv venv -q --python 3.11 /opt/s2c/py && uv pip install -q --python /opt/s2c/py/bin/python -r /tmp/worker.txt
RUN uv venv -q --python 3.11 /opt/s2c/teacher \
    && uv pip install -q --python /opt/s2c/teacher/bin/python -r /tmp/teacher.txt

FROM ubuntu:22.04 AS light
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=envs /opt/s2c /opt/s2c

FROM ${ISAAC_IMAGE} AS isaac
COPY --from=envs /opt/s2c /opt/s2c
