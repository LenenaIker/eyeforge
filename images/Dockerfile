# The worker's images and Python kits (docs/imagenes.md), built and published by images/build.sh:
#   light  for runs without synthetic data: Ubuntu plus the tools to unpack the kits
#   isaac  NVIDIA's Isaac Sim image plus the same tools, for runs that render
#   kits   the Pythons, TensorFlow (student), torch (teachers) and PaddlePaddle (PicoDet counter, with the
#          PaddleDetection checkout and its COCO weights), as zstd tarballs for the cache bucket
# The Pythons are not inside the images: a run pulls only the kits it uses (the teachers' only when it labels
# photos, Paddle's only when it trains a PicoDet counter), streamed from the bucket into tar while the others
# download (docs/imagenes.md). They live in /tmp/s2c/envs, a path every image can write (Isaac's runs as a
# normal user), so they unpack where they were built. No weights either: the teachers' come from the cache bucket (sim2chip.teacher.weights), the
# shader caches too (sim2chip.cloud_worker). The pip lists are written by build.sh from
# sim2chip/backends/nebius.py (WORKER_DEPS, WORKER_NODEPS, TEACHER_DEPS,
# PICODET_DEPS, PICODET_NODEPS, PADDLEDET_REF), the one place they live.
ARG ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0

FROM ubuntu:22.04 AS envs
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/tmp/s2c/envs/pythons UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates zstd curl \
    && rm -rf /var/lib/apt/lists/*
# One COPY per kit, right before it: a changed list rebuilds only its own kit.
COPY worker.txt worker-nodeps.txt /tmp/
RUN uv venv -q --python 3.11 /tmp/s2c/envs/py && uv pip install -q --python /tmp/s2c/envs/py/bin/python -r /tmp/worker.txt \
    && uv pip install -q --no-deps --python /tmp/s2c/envs/py/bin/python -r /tmp/worker-nodeps.txt
COPY teacher.txt /tmp/
RUN uv venv -q --python 3.11 /tmp/s2c/envs/teacher \
    && uv pip install -q --python /tmp/s2c/envs/teacher/bin/python -r /tmp/teacher.txt
COPY picodet.txt picodet-nodeps.txt paddledet.txt /tmp/
RUN uv venv -q --python 3.11 /tmp/s2c/envs/picodet \
    && UV_HTTP_TIMEOUT=300 uv pip install -q --python /tmp/s2c/envs/picodet/bin/python -r /tmp/picodet.txt \
    && uv pip install -q --no-deps --python /tmp/s2c/envs/picodet/bin/python -r /tmp/picodet-nodeps.txt
RUN mkdir -p /tmp/s2c/envs/paddledet/weights \
    && curl -fsSL "https://github.com/PaddlePaddle/PaddleDetection/archive/refs/heads/$(cat /tmp/paddledet.txt).tar.gz" \
       | tar xzf - -C /tmp/s2c/envs/paddledet --strip-components 1 \
    && curl -fsSL -o /tmp/s2c/envs/paddledet/weights/picodet_s_320_coco_lcnet.pdparams \
       https://paddledet.bj.bcebos.com/models/picodet_s_320_coco_lcnet.pdparams
RUN mkdir /kits && cd /tmp/s2c/envs \
    && for d in pythons py teacher; do tar cf - "$d" | zstd -T0 -3 -q -o "/kits/$d.tar.zst"; done \
    && tar cf - picodet paddledet | zstd -T0 -3 -q -o /kits/picodet.tar.zst

FROM scratch AS kits
COPY --from=envs /kits /

FROM ubuntu:22.04 AS light
# libgomp1: PaddlePaddle (picodet kit) needs OpenMP, which a bare Ubuntu lacks.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates zstd aria2 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM ${ISAAC_IMAGE} AS isaac
USER root
RUN apt-get update && apt-get install -y --no-install-recommends zstd aria2 && rm -rf /var/lib/apt/lists/*
USER isaac-sim
