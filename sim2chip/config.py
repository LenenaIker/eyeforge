"""Settings from the environment, falling back to the repo's .env (gitignored; never printed).

    env("SIM2CHIP_REGION", "eu-north1")      text, or the default
    env_int("SIM2CHIP_WARM_HOURS", 2)        a number; a value that is not one stops with the setting's name
    check()                                  what is wrong with the settings, for `sim2chip serve` and `config`

Every SIM2CHIP_* and S2C_* name the code reads is in KNOWN (tests/test_config.py keeps it so), so a name that is
not there is a typo whose value nothing reads. What each one does is in .env.example."""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

INT, FLOAT = "int", "float"
# name -> its kind when it is a number, else None. The S2C_* ones are set by the server for its cloud workers.
KNOWN: dict[str, str | None] = {
    # where things are
    "SIM2CHIP_HOME": None, "SIM2CHIP_LOCKS": None, "SIM2CHIP_TOOLS": None, "SIM2CHIP_WEIGHTS_DIR": None,
    "SIM2CHIP_TEACHER_SOCKET": None, "SIM2CHIP_PADDLEDET": None, "SIM2CHIP_TRELLIS_DIR": None,
    "SIM2CHIP_GRUT_DIR": None,
    # interpreters of the parts that need their own environment
    "SIM2CHIP_PYTHON": None, "SIM2CHIP_ISAAC_PYTHON": None, "SIM2CHIP_SAM3_PYTHON": None,
    "SIM2CHIP_PICODET_PYTHON": None, "SIM2CHIP_TRELLIS_PYTHON": None, "SIM2CHIP_ROUTE_PYTHON": None,
    "SIM2CHIP_SCENE_PYTHON": None, "SIM2CHIP_ERASE_PYTHON": None, "SIM2CHIP_MAPANYTHING_PYTHON": None,
    "SIM2CHIP_GSPLAT_PYTHON": None,
    # models and the pipeline
    "SIM2CHIP_LLM_BASE_URL": None, "SIM2CHIP_LLM_WORKERS": INT, "SIM2CHIP_LLM_DAILY_USD": FLOAT,
    "SIM2CHIP_LOCAL_VLM_URL": None, "SIM2CHIP_LOCAL_VLM_IDLE": FLOAT, "SIM2CHIP_LOCAL_VLM_MAX_TOKENS": INT,
    "SIM2CHIP_PARALLEL_SEEDS": INT, "SIM2CHIP_PICODET_STEPS": INT, "SIM2CHIP_LABEL_SAMPLE": INT,
    "SIM2CHIP_LIKENESS": None, "SIM2CHIP_SCENE_VIEWS": INT, "SIM2CHIP_WALK_SCENES": None, "SIM2CHIP_RUNS_ON": None,
    "SIM2CHIP_QWEN_NF4": None, "SIM2CHIP_ERASE_BITS": INT, "SIM2CHIP_ERASE_GPU": None,
    # Nebius
    "SIM2CHIP_NEBIUS_CLI": None, "SIM2CHIP_NEBIUS_PROJECT": None, "SIM2CHIP_NEBIUS_REGISTRY": None,
    "SIM2CHIP_NEBIUS_SECRET": None, "SIM2CHIP_NEBIUS_MODE": None, "SIM2CHIP_NEBIUS_PLATFORM": None,
    "SIM2CHIP_NEBIUS_PRESET": None, "SIM2CHIP_NEBIUS_FALLBACK": None, "SIM2CHIP_NEBIUS_PROVISION_S": FLOAT,
    "SIM2CHIP_NEBIUS_ELSEWHERE": None, "SIM2CHIP_NEBIUS_PROJECTS": None, "SIM2CHIP_SM120_KITS": None,
    "SIM2CHIP_IMAGE_TAG": None, "SIM2CHIP_REGION": None, "SIM2CHIP_S3_SECRET": None,
    "SIM2CHIP_BUCKET_PREFIX": None, "SIM2CHIP_CACHE_BUCKET": None, "SIM2CHIP_GPU_NAME": None,
    "SIM2CHIP_USD_PER_HOUR": FLOAT, "SIM2CHIP_MAX_USD": FLOAT, "SIM2CHIP_DAILY_USD": FLOAT,
    "SIM2CHIP_WORKER_HOURS": INT, "SIM2CHIP_WS_PROJECT": None, "SIM2CHIP_WS_SUBNET": None,
    # the warm GPU and the scene GPU
    "SIM2CHIP_WARM": None, "SIM2CHIP_WARM_HOURS": INT, "SIM2CHIP_WARM_IDLE_S": FLOAT, "SIM2CHIP_WARM_KEEP_S": FLOAT,
    "SIM2CHIP_WARM_SILENT_S": FLOAT, "SIM2CHIP_WARM_SETUP_S": FLOAT, "SIM2CHIP_RUN_RAM_GB": FLOAT,
    "SIM2CHIP_SCENE_QUEUE": None, "SIM2CHIP_SCENE_PARALLEL": INT, "SIM2CHIP_SCENE_GPU": None,
    "SIM2CHIP_SCENE_BIG": None,
    # the public server
    "SIM2CHIP_WEB_KEY": None, "SIM2CHIP_PUBLIC_URL": None, "SIM2CHIP_SESSION_SECRET": None,
    "SIM2CHIP_CONTACT": None, "SIM2CHIP_JUDGE_UNTIL": None, "SIM2CHIP_STRIPE_TAX": None,
    # set by the server for its cloud workers
    "S2C_JOB_BUCKET": None, "S2C_CACHE_BUCKET": None, "S2C_CODE_URL": None, "S2C_ENVS": None,
    "S2C_ENVS_LATE": None, "S2C_SETUP_S": FLOAT, "S2C_T0": FLOAT, "S2C_WARM": None, "S2C_WARM_ID": None,
    "S2C_WARM_HOURS": FLOAT, "S2C_WARM_PARALLEL": INT,
}
FAMILIES = [re.compile(r"SIM2CHIP_[A-Z0-9]+_MODEL")]  # the model of a role (llm.Client.model_for)
_dotenv_cache: dict[Path, tuple[float, dict[str, str]]] = {}


class ConfigError(ValueError):
    """A setting whose value cannot be what it is for."""


def _dotenv() -> dict[str, str]:
    """The repo's .env as a dict, read again only when the file changes."""
    f = REPO / ".env"
    try:
        mtime = f.stat().st_mtime
    except OSError:
        return {}
    if f not in _dotenv_cache or _dotenv_cache[f][0] != mtime:
        values = {}
        for line in f.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
            if m and m.group(1) not in values:  # the first one wins, as before
                values[m.group(1)] = m.group(2).strip().strip('"').strip("'")
        _dotenv_cache[f] = (mtime, values)
    return _dotenv_cache[f][1]


def env(name: str, default: str | None = None) -> str | None:
    if name in os.environ:
        return os.environ[name]
    return _dotenv().get(name, default)


def _number(name: str, default, kind):
    raw = env(name)
    if raw is None or not raw.strip():
        return default
    try:
        return kind(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a {'whole number' if kind is int else 'number'}") from None


def env_int(name: str, default: int | None) -> int | None:
    return _number(name, default, int)


def env_float(name: str, default: float | None) -> float | None:
    return _number(name, default, float)


def known(name: str) -> bool:
    return name in KNOWN or any(f.fullmatch(name) for f in FAMILIES)


def check() -> list[str]:
    """What is wrong with the settings, without their values: names nothing reads (a typo leaves the default in
    use), numbers that are not numbers, interpreters and folders that are not there."""
    names = {n for n in [*os.environ, *_dotenv()] if n.startswith(("SIM2CHIP_", "S2C_"))}
    out = [f"{n} is not a setting EyeForge reads (a typo?)" for n in sorted(names) if not known(n)]
    for n in sorted(names & set(KNOWN)):
        value = env(n) or ""
        if KNOWN[n] is not None:
            try:
                _number(n, None, int if KNOWN[n] == INT else float)
            except ConfigError as e:
                out.append(str(e))
        elif value and n.endswith("_PYTHON") and not Path(value).is_file():
            out.append(f"{n} names an interpreter that is not there")
        elif value and (n.endswith("_DIR") or n in ("SIM2CHIP_TOOLS", "SIM2CHIP_PADDLEDET")) and not Path(value).is_dir():
            out.append(f"{n} names a folder that is not there")
    return out


def gpu_name() -> str:
    """The cloud GPU, for people (logs and the web); goes with SIM2CHIP_NEBIUS_PLATFORM."""
    return env("SIM2CHIP_GPU_NAME", "L40S")
