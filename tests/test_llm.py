"""Token Factory client against a local fake OpenAI-compatible server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sim2chip.llm import Client, LLMError, image_part, parse_json, usage_summary


class Fake(BaseHTTPRequestHandler):
    calls = []
    fail_first = 0

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"data": [{"id": "meta/llama"}, {"id": "nvidia/nemotron-3-super-120b-a12b"},
                                  {"id": "nvidia/Nemotron-3-Nano-Omni"}, {"id": "zai-org/GLM-5.3-Flash"},
                                  {"id": "openbmb/MiniCPM-V-4_5"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.calls.append((self.headers["Authorization"], body))
        if Fake.fail_first > 0:
            Fake.fail_first -= 1
            return self._send(503, {"error": "busy"})
        self._send(200, {"choices": [{"message": {"content": '<think>hmm</think>```json\n{"ok": true}\n```'}}],
                         "usage": {"prompt_tokens": 11, "completion_tokens": 5}})


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Fake.calls.clear()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_chat_logs_usage(server, tmp_path):
    log = tmp_path / "usage.jsonl"
    c = Client(base_url=server, key="k", usage_log=log, retries=0)
    text = c.chat([{"role": "user", "content": "hi"}], "m", purpose="qa")
    assert parse_json(text) == {"ok": True}
    assert Fake.calls[0][0] == "Bearer k"
    s = usage_summary(log)["qa"]
    assert s["calls"] == 1 and s["prompt_tokens"] == 11 and s["models"] == ["m"]


def test_retries_on_503(server):
    Fake.fail_first = 1
    c = Client(base_url=server, key="k", retries=2)
    c.chat([{"role": "user", "content": "hi"}], "m", purpose="x")
    assert len(Fake.calls) == 2


def test_error_without_retries(server):
    Fake.fail_first = 1
    with pytest.raises(LLMError, match="503"):
        Client(base_url=server, key="k", retries=0).chat([], "m", purpose="x")


def test_model_selection(server, monkeypatch):
    for role in ("VISION", "DESCRIBE"):
        monkeypatch.delenv(f"SIM2CHIP_{role}_MODEL", raising=False)
    c = Client(base_url=server, key="k")
    assert c.model_for("text") == "nvidia/nemotron-3-super-120b-a12b"
    assert c.model_for("vision") == "nvidia/Nemotron-3-Nano-Omni"
    c._models = [m for m in c.models() if "Omni" not in m]
    assert c.model_for("vision") == "zai-org/GLM-5.3-Flash"
    assert c.model_for("describe") == "openbmb/MiniCPM-V-4_5"
    monkeypatch.setenv("SIM2CHIP_VISION_MODEL", "x/y")
    assert c.model_for("vision") == "x/y"
    # The interviewer: Nemotron 3.5 Super VL when listed, else Super.
    for role in ("INTERVIEW", "SCENE"):
        monkeypatch.delenv(f"SIM2CHIP_{role}_MODEL", raising=False)
    assert c.model_for("interview") == "nvidia/nemotron-3-super-120b-a12b"
    c._models = c.models() + ["nvidia/Nemotron-3.5-Super-VL"]
    assert c.model_for("interview") == "nvidia/Nemotron-3.5-Super-VL"
    assert c.model_for("scene") == "nvidia/Nemotron-3.5-Super-VL"


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    assert parse_json('{"say": "hi"}\n{"say": "again"}') == {"say": "hi"}
    assert parse_json('{"say": "a {b}"} then {oops') == {"say": "a {b}"}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_image_part(tmp_path):
    f = tmp_path / "x.png"
    f.write_bytes(b"\x89PNG")
    part = image_part(f)
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_map_within_keeps_what_answered_in_time():
    import threading
    import time

    from sim2chip.llm import map_within
    release = threading.Event()

    def look(n):
        if n == 1:
            raise RuntimeError("model error")
        if n == 2:
            release.wait(5)  # a photo the model never answers in time
        return n * 10
    seen = []
    t0 = time.time()
    out = map_within(look, [0, 1, 2, 3], seconds=0.3, workers=4, progress=lambda d, t: seen.append((d, t)))
    release.set()
    assert out == [0, None, None, 30] and time.time() - t0 < 2
    assert seen[-1] == (3, 4)  # the failed one counts as looked at, the late one does not
