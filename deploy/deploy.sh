#!/usr/bin/env bash
# Build the EyeForge image on this machine and run it on the demo server (docs/deploy.md).
# The server is small: building there would take its memory from the other site, so the image travels.
#   deploy/deploy.sh                 # EYEFORGE_HOST=root@<server> to use another one
set -euo pipefail

HOST=${EYEFORGE_HOST:-root@<server-ip>}
DIR=/opt/eyeforge
cd "$(dirname "$0")/.."

ssh "$HOST" "test -f $DIR/.env" || {
  echo "no $DIR/.env on $HOST: fill in deploy/env.example there first" >&2
  exit 1
}

docker build --platform linux/amd64 -t eyeforge:latest .
docker save eyeforge:latest | gzip | ssh "$HOST" 'gunzip | docker load'
rsync -a deploy/docker-compose.yml deploy/backup.sh "$HOST:$DIR/"
ssh "$HOST" "set -e
  mkdir -p $DIR/data $DIR/nebius
  docker network inspect eyeforge >/dev/null 2>&1 || docker network create eyeforge
  echo '15 4 * * * root $DIR/backup.sh >> $DIR/data/backups/backup.log 2>&1' > /etc/cron.d/eyeforge-backup
  cd $DIR && docker compose up -d
  docker image prune -f >/dev/null"
echo "EyeForge is up on $HOST; logs: ssh $HOST docker logs -f eyeforge"
