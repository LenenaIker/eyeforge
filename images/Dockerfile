# The worker's images and Python kits (docs/imagenes.md), built and published by images/build.sh:
#   light  for runs without synthetic data: Ubuntu plus the tools to unpack the kits
#   isaac  NVIDIA's Isaac Sim image plus the same tools, for runs that render
#   kits   the two Pythons, TensorFlow (student) and torch (teachers), as zstd tarballs for the cache bucket
# The Pythons are not inside the images: a run pulls only the kits it uses (the teachers' only when it labels
# photos), streamed from the bucket into tar while the others download (docs/imagenes.md). They live
# in /tmp/s2c/envs, a path every image can write (Isaac's runs as a normal user), so they unpack where they
# were built. No weights either: the teachers' come from the cache bucket (sim2chip.teacher.weights), the
# shader caches too (sim2chip.cloud_worker). The pip lists are written by build.sh from
# sim2chip/backends/nebius.py (WORKER_DEPS, TEACHER_DEPS), the one place they live.
ARG ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0

FROM ubuntu:22.04 AS envs
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/tmp/s2c/envs/pythons UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates zstd && rm -rf /var/lib/apt/lists/*
COPY worker.txt teacher.txt /tmp/
RUN uv venv -q --python 3.11 /tmp/s2c/envs/py && uv pip install -q --python /tmp/s2c/envs/py/bin/python -r /tmp/worker.txt
RUN uv venv -q --python 3.11 /tmp/s2c/envs/teacher \
    && uv pip install -q --python /tmp/s2c/envs/teacher/bin/python -r /tmp/teacher.txt
RUN mkdir /kits && cd /tmp/s2c/envs \
    && for d in pythons py teacher; do tar cf - "$d" | zstd -T0 -3 -q -o "/kits/$d.tar.zst"; done

FROM scratch AS kits
COPY --from=envs /kits /

FROM ubuntu:22.04 AS light
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates zstd aria2 \
    && rm -rf /var/lib/apt/lists/*

FROM ${ISAAC_IMAGE} AS isaac
USER root
RUN apt-get update && apt-get install -y --no-install-recommends zstd aria2 && rm -rf /var/lib/apt/lists/*
USER isaac-sim
