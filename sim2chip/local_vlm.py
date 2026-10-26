"""A local vision model behind the same chat API as Token Factory, so the scene checks run on this machine's GPU
for free (user, 2026-10-15: Qwen3.8-27B on Token Factory cost 47 $, mostly reasoning tokens). The client sends
every model named "local/<hf id>" here (llm.Client); SIM2CHIP_REVIEW_MODEL=local/nvidia/Cosmos-Reason2-8B moves
the vision checks over.

The model loads at the first call and is dropped after SIM2CHIP_LOCAL_VLM_IDLE seconds without calls (default
60), or at once on POST /v1/unload (a caller runs it before a step that needs the GPU: Cosmos takes 18 of the
3090's 24 GB), so Isaac and the other GPU steps get the memory back. One answer at a time (one GPU). Run in
the `teacher-sam3` env (transformers with Qwen3-VL):

    python -m sim2chip.local_vlm [--port 8765]

Configuration (environment or .env): SIM2CHIP_LOCAL_VLM_URL (the client's side, default
http://127.0.0.1:8765/v1), SIM2CHIP_LOCAL_VLM_MAX_TOKENS (answer cap, default 2048), HF_HUB_CACHE.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import env, env_float, env_int

DEFAULT_MODEL = "nvidia/Cosmos-Reason2-8B"
IDLE_S = env_float("SIM2CHIP_LOCAL_VLM_IDLE", 60)
MAX_TOKENS = env_int("SIM2CHIP_LOCAL_VLM_MAX_TOKENS", 2048)  # observe.json's answers run past 1024


class Model:
    """The loaded model, or none: loaded on demand, dropped when idle; one generation at a time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.name = None
        self.model = self.proc = None
        self.last = time.time()
        threading.Thread(target=self._reaper, daemon=True).start()

    def _load(self, name: str):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if self.name == name:
            return
        self._drop()
        t0 = time.time()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
        self.proc = AutoProcessor.from_pretrained(name)
        self.name = name
        print(f"loaded {name} in {time.time() - t0:.0f} s", flush=True)

    def _drop(self):
        if self.model is None:
            return
        import gc

        import torch
        self.model = self.proc = self.name = None
        gc.collect()
        torch.cuda.empty_cache()
        print("dropped the model (idle)", flush=True)

    def _reaper(self):
        while True:
            time.sleep(5)
            if self.model is not None and time.time() - self.last > IDLE_S and self.lock.acquire(blocking=False):
                try:
                    if time.time() - self.last > IDLE_S:
                        self._drop()
                finally:
                    self.lock.release()

    def chat(self, name: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        import torch
        with self.lock:
            self._load(name)
            inp = self.proc.apply_chat_template(to_qwen(messages), tokenize=True, add_generation_prompt=True,
                                                return_dict=True, return_tensors="pt").to("cuda")
            sample = temperature > 0
            with torch.inference_mode():
                out = self.model.generate(**inp, max_new_tokens=min(max_tokens, MAX_TOKENS), do_sample=sample,
                                          **({"temperature": temperature} if sample else {}))
            new = out[0, inp["input_ids"].shape[1]:]
            text = self.proc.decode(new, skip_special_tokens=True)
            self.last = time.time()
            return {"text": text, "prompt_tokens": int(inp["input_ids"].shape[1]), "completion_tokens": int(len(new))}


def to_qwen(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages (text, data-URI images) as the Qwen3-VL processor takes them (PIL images)."""
    from PIL import Image
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            parts = []
            for p in content:
                if p.get("type") == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    data = base64.b64decode(url.split(",", 1)[1])
                    parts.append({"type": "image", "image": Image.open(io.BytesIO(data)).convert("RGB")})
        out.append({"role": m["role"], "content": parts})
    return out


MODEL = Model()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": f"local/{DEFAULT_MODEL}"}]})
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):
        if self.path.rstrip("/").endswith("/unload"):  # a GPU step is about to start (photo2scene.Flow.run)
            with MODEL.lock:
                MODEL._drop()
            self._send(200, {"unloaded": True})
            return
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"detail": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            name = body["model"].removeprefix("local/")
            t0 = time.time()
            r = MODEL.chat(name, body["messages"], int(body.get("max_tokens") or MAX_TOKENS),
                           float(body.get("temperature") or 0.0))
        except Exception as e:  # the client sees why (a bad request, or the GPU out of memory)
            self._send(500, {"detail": f"{type(e).__name__}: {e}"[:500]})
            return
        print(f"{name}: {r['prompt_tokens']} + {r['completion_tokens']} tokens in {time.time() - t0:.1f} s", flush=True)
        self._send(200, {"object": "chat.completion", "model": body["model"],
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"]}})

    def log_message(self, *a):  # one line per answer from do_POST instead
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if env("HF_HUB_CACHE"):
        os.environ.setdefault("HF_HUB_CACHE", env("HF_HUB_CACHE"))
    print(f"local vision model on 127.0.0.1:{args.port} (loads at the first call, drops after {IDLE_S:.0f} s idle)",
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
