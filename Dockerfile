# EyeForge web server for a public demo: the interview, uploads and results. The GPU work runs in Nebius
# AI Cloud (--backend nebius), so this image needs no GPU, Isaac Sim or TensorFlow.
#   docker build -t eyeforge .
#   docker run -p 8000:8000 --env-file .env -v eyeforge-data:/data -v ~/.nebius:/root/.nebius eyeforge
# The nebius CLI inside uses the mounted ~/.nebius profile (a service account on a server). The CLI itself
# lives in /usr/local/bin, so that mount does not hide it. The demo server's setup is in deploy/.

# The web app (web/, React): built here, served by the Python server from sim2chip/web/dist.
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web ./
RUN npm run build -- --outDir /dist

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash \
    && mv /root/.nebius/bin/nebius /usr/local/bin/nebius && rm -rf /root/.nebius

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY sim2chip sim2chip
COPY --from=web /dist sim2chip/web/dist
# Editable, so the package finds its repo folder (the code archive sent to the cloud jobs is built from it).
RUN pip install --no-cache-dir -e ".[web,cloud,assets]" pillow

# Jobs, uploads, the asset cache and spend records: keep them on a volume.
ENV SIM2CHIP_HOME=/data
VOLUME /data
EXPOSE 8000
CMD ["sim2chip", "serve", "--backend", "nebius", "--public", "--host", "0.0.0.0"]
