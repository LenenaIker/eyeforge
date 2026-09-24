import pytest


@pytest.fixture(autouse=True)
def no_network_in_prompts(monkeypatch):
    """The interview prompt lists Poly Haven tags live; tests use the offline fallback."""
    from sim2chip import interview
    monkeypatch.setattr(interview, "scene_tags", lambda kind: interview.FALLBACK_TAGS[kind])
