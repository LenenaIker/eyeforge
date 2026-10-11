import pytest


@pytest.fixture(autouse=True)
def no_network_in_prompts(monkeypatch):
    """The interview prompt lists Poly Haven tags live; tests use the offline fallback."""
    from sim2chip import interview
    monkeypatch.setattr(interview, "scene_tags", lambda kind: interview.FALLBACK_TAGS[kind])


@pytest.fixture
def template_on(monkeypatch):
    """The scene template and its 3D model stages, off in the product since 2026-10-11 (stages.base.TEMPLATE), for
    the tests of the pipeline's machinery that still use them as their example until the code goes (phase C)."""
    from sim2chip.stages import base
    monkeypatch.setattr(base, "TEMPLATE", True)
