"""Execution backends: where and how each stage runs."""
from __future__ import annotations


def get(name: str, verbose: bool = False, **kw):
    """Factory: the backend by name, imported only when asked for (patterns: docs/architecture.md)."""
    if name == "local":
        from .local import LocalBackend
        return LocalBackend(verbose=verbose)
    if name == "nebius":
        from .nebius import NebiusBackend
        return NebiusBackend(verbose=verbose, **kw)
    raise KeyError(f"unknown backend '{name}', known: local, nebius")
