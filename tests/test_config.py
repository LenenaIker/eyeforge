"""Settings (sim2chip/config.py): .env reaches every setting, a bad number names its setting, and check() finds
typos, bad numbers and missing interpreters without printing a value."""
import ast
import re
from pathlib import Path

import pytest

from sim2chip import config

ROOT = Path(__file__).resolve().parents[1]
CODE = [*(ROOT / "sim2chip").rglob("*.py"), *(ROOT / "experiments" / "worldstereo" / "wsroute").glob("*.py"),
        ROOT / "experiments" / "worldstereo" / "ws_route.py"]


@pytest.fixture
def dotenv(tmp_path, monkeypatch):
    """A repo whose .env the test writes, and an environment without our settings."""
    monkeypatch.setattr(config, "REPO", tmp_path)
    for name in list(config.os.environ):
        if name.startswith(("SIM2CHIP_", "S2C_")):
            monkeypatch.delenv(name)
    return tmp_path / ".env"


def test_every_setting_the_code_reads_is_known():
    names = set()
    for f in CODE:
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and "#define" not in n.value:  # C macros
                names |= set(re.findall(r"\b(?:SIM2CHIP|S2C)_[A-Z0-9_]*[A-Z0-9]\b", n.value))
    assert sorted(n for n in names if not config.known(n)) == []


def test_dotenv_reaches_settings_and_the_environment_wins(dotenv, monkeypatch):
    dotenv.write_text('# a comment\nSIM2CHIP_PARALLEL_SEEDS=5\nexport SIM2CHIP_REGION="eu-west1"\n')
    assert config.env_int("SIM2CHIP_PARALLEL_SEEDS", 3) == 5
    assert config.env("SIM2CHIP_REGION") == "eu-west1"
    monkeypatch.setenv("SIM2CHIP_PARALLEL_SEEDS", "2")
    assert config.env_int("SIM2CHIP_PARALLEL_SEEDS", 3) == 2
    assert config.env_int("SIM2CHIP_LABEL_SAMPLE", 0) == 0 and config.env_float("SIM2CHIP_LLM_DAILY_USD", None) is None


def test_a_changed_dotenv_is_read_again(dotenv):
    dotenv.write_text("SIM2CHIP_REGION=a\n")
    assert config.env("SIM2CHIP_REGION") == "a"
    dotenv.write_text("SIM2CHIP_REGION=b\n")
    import os
    os.utime(dotenv, (1, 1))  # another mtime, as an edit a second later has
    assert config.env("SIM2CHIP_REGION") == "b"


def test_a_bad_number_names_its_setting(dotenv, monkeypatch):
    monkeypatch.setenv("SIM2CHIP_WARM_HOURS", "two")
    with pytest.raises(config.ConfigError, match="SIM2CHIP_WARM_HOURS must be a whole number"):
        config.env_int("SIM2CHIP_WARM_HOURS", 2)


def test_check_finds_typos_bad_numbers_and_missing_interpreters_without_values(dotenv, monkeypatch, tmp_path):
    dotenv.write_text("SIM2CHIP_WARM_HOUR=secret-ish\nSIM2CHIP_USD_PER_HOUR=cheap\n")
    monkeypatch.setenv("SIM2CHIP_SAM3_PYTHON", str(tmp_path / "nope" / "python"))
    monkeypatch.setenv("SIM2CHIP_REVIEW_MODEL", "local/x")  # a role's model: a family, not a typo
    problems = config.check()
    text = "\n".join(problems)
    assert "SIM2CHIP_WARM_HOUR is not a setting" in text
    assert "SIM2CHIP_USD_PER_HOUR must be a number" in text
    assert "SIM2CHIP_SAM3_PYTHON names an interpreter that is not there" in text
    assert "REVIEW_MODEL" not in text and "secret-ish" not in text and "cheap" not in text
