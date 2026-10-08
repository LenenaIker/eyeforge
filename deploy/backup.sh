#!/usr/bin/env bash
# Daily copy of the demo server's databases (docs/deploy.md): accounts.db holds balances and payments,
# library.db what users keep. SQLite's online backup, run by the container's Python, is safe while the
# server writes. Keeps KEEP days in /opt/eyeforge/data/backups. Run by /etc/cron.d/eyeforge-backup.
set -euo pipefail

KEEP=${KEEP:-7}
DIR=/opt/eyeforge/data/backups
DAY=$(date +%F)
mkdir -p "$DIR"
docker exec -i eyeforge python - "$DAY" <<'PY'
import sqlite3, sys
from pathlib import Path
day = sys.argv[1]
for src in (Path("/data/accounts.db"), Path("/data/library/library.db")):
    if not src.exists():
        continue
    dst = Path("/data/backups") / f"{src.stem}-{day}.db"
    with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
        s.backup(d)
    with sqlite3.connect(dst) as d:
        assert d.execute("PRAGMA integrity_check").fetchone()[0] == "ok", dst
    print(f"{dst.name}: {dst.stat().st_size} bytes")
PY
gzip -f "$DIR"/*-"$DAY".db
chmod 600 "$DIR"/*.gz
find "$DIR" -name '*.db.gz' -mtime +"$KEEP" -delete
