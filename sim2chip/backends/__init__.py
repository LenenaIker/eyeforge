"""Execution backends: where and how each stage runs."""
from __future__ import annotations


def get(name: str, verbose: bool = False):
    if name == "local":
        from .local import LocalBackend
        return LocalBackend(verbose=verbose)
    if name == "nebius":
        raise NotImplementedError("the Nebius backend (Serverless AI jobs) is planned for week 3")
    raise KeyError(f"unknown backend '{name}', known: local, nebius")
