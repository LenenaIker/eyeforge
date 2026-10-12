#!/usr/bin/env bash
# Copies the built firmware (pio run) into the web app, which flashes it from the browser, and writes the
# firmware version (FIRMWARE_VERSION in src/main.cpp) and build date into the web's manifest.json.
#   cd firmware/esp32cam && pio run && tools/copy_web.sh
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
build="$here/.pio/build/esp32cam"
web="$here/../../web/public/firmware/esp32"
version="$(sed -n 's/^#define FIRMWARE_VERSION "\([^"]*\)".*/\1/p' "$here/src/main.cpp")"
[ -n "$version" ] || { echo "FIRMWARE_VERSION not found in src/main.cpp" >&2; exit 1; }
for f in bootloader.bin partitions.bin ota_data_initial.bin firmware.bin; do
  [ -f "$build/$f" ] || { echo "missing $build/$f: run pio run first" >&2; exit 1; }
  cp "$build/$f" "$web/$f"
done
python3 - "$web/manifest.json" "$version" "$(date +%F)" <<'PY'
import re, sys
path, version, day = sys.argv[1:]
text = open(path).read()
text = re.sub(r'"build": "[^"]*"', f'"build": "firmware/esp32cam, ESP-IDF, built {day}"', text)
if '"firmware":' in text:
    text = re.sub(r'"firmware": "[^"]*"', f'"firmware": "{version}"', text)
else:
    text = text.replace('"board":', f'"firmware": "{version}",\n  "board":', 1)
open(path, "w").write(text)
PY
echo "copied firmware $version to $web"
