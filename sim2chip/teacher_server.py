"""The teachers kept loaded on the warm GPU, in the teachers' Python (user, 2026-10-09):

    python -m sim2chip.teacher_server --socket /tmp/s2c/teacher.sock --preload sam3=<weights> siglip=<weights>

A task that labels photos used to start a new process per teacher and load its weights each time: about 54 s of
a 6-photo calibration round were mostly that. This process loads SAM 3 and SigLIP 2 once, when the warm GPU
starts, and answers on a Unix socket. teacher.py uses it when SIM2CHIP_TEACHER_SOCKET points to it, else runs the
workers as before.

Protocol: one JSON line {"v": VERSION, "teacher": "sam3" | "siglip", "req": <the worker's request>}; back come the
worker's progress lines, then DONE + "0" or DONE + "1 <error>". One request at a time on the GPU.
"""
from __future__ import annotations

import argparse
import importlib
import json
import socketserver
import sys
import threading
import time
import traceback
from pathlib import Path

VERSION = 1
DONE = "\x00done "
WORKERS = {"sam3": "sim2chip.sam3_worker", "siglip": "sim2chip.siglip_worker"}

_loaded: dict[tuple[str, str], object] = {}
# One load at a time, of any teacher: two at once import transformers from two threads, and its lazy imports are
# not thread-safe ("cannot import name 'Sam3Model'" on the warm GPU, 2026-10-09).
_loading = threading.Lock()
_gpu = threading.Lock()  # one request at a time; loading takes no turn, so SigLIP answers while SAM 3 loads


def loaded(teacher: str, model: str):
    key = (teacher, model)
    if key in _loaded:
        return _loaded[key]
    with _loading:
        if key not in _loaded:
            _loaded[key] = importlib.import_module(WORKERS[teacher]).load(model)
    return _loaded[key]


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        def say(line: str) -> None:
            self.wfile.write((line.rstrip("\n") + "\n").encode())
            self.wfile.flush()

        try:
            ask = json.loads(self.rfile.readline())
            if ask.get("v") != VERSION or ask.get("teacher") not in WORKERS:
                raise ValueError(f"unknown request (version {ask.get('v')}, teacher {ask.get('teacher')})")
            req = ask["req"]
            model = loaded(ask["teacher"], req["model"])
            with _gpu:
                importlib.import_module(WORKERS[ask["teacher"]]).run(req, model, say)
            say(DONE + "0")
        except Exception as e:
            traceback.print_exc()
            try:
                say(DONE + f"1 {type(e).__name__}: {e}"[:500])
            except OSError:  # the task is gone
                pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--socket", required=True)
    p.add_argument("--preload", nargs="*", default=[], help="teacher=weights folder, loaded before any request")
    args = p.parse_args()
    sock = Path(args.socket)
    sock.unlink(missing_ok=True)
    server = socketserver.ThreadingUnixStreamServer(str(sock), Handler)

    def preload() -> None:
        for item in args.preload:
            teacher, model = item.split("=", 1)
            t0 = time.time()
            try:
                loaded(teacher, model)
                print(f"teacher server: {teacher} loaded in {time.time() - t0:.0f} s", flush=True)
            except Exception as e:  # a request loads it again and says why
                print(f"teacher server: {teacher} not loaded: {e}", flush=True)

    threading.Thread(target=preload, daemon=True).start()
    print(f"teacher server on {sock}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
