"""Settings from the environment, falling back to the repo's .env (gitignored; never printed)."""
from __future__ import annotations

import os

from .workspace import REPO


def env(name: str, default: str | None = None) -> str | None:
    if name in os.environ:
        return os.environ[name]
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default


def gpu_name() -> str:
    """The cloud GPU, for people (logs and the web); goes with SIM2CHIP_NEBIUS_PLATFORM."""
    return env("SIM2CHIP_GPU_NAME", "L40S")
