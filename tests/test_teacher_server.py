import sys
import threading
import time
import types

from sim2chip import teacher_server


def test_teachers_load_one_at_a_time(monkeypatch):
    """Two teachers loading at once imported transformers from two threads and one failed (2026-10-09)."""
    busy, overlaps = [], []

    def load(model):
        busy.append(model)
        overlaps.append(len(busy))
        time.sleep(0.05)
        busy.remove(model)
        return model
    for name in ("fake_a", "fake_b"):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(load=load))
    monkeypatch.setattr(teacher_server, "WORKERS", {"a": "fake_a", "b": "fake_b"})
    monkeypatch.setattr(teacher_server, "_loaded", {})
    threads = [threading.Thread(target=teacher_server.loaded, args=(t, f"w{t}")) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max(overlaps) == 1 and teacher_server.loaded("a", "wa") == "wa"
