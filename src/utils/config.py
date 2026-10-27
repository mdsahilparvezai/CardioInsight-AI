"""Configuration loading for CardioInsight AI.

Loads ``configs/config.yaml`` and overlays values from environment variables
prefixed with ``CARDIOINSIGHT_``.  Nothing here is secret: the environment
overrides exist so deployment-specific paths (data location, tracking URI)
can be set without editing source files.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - PyYAML is a hard dependency
    yaml = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Module-level cache.  ``get_config()`` reads it; ``reset_config()`` clears it
# so tests can load a different config in the same process.
_CACHED_CONFIG: dict | None = None


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into ``base`` (returns a new dict)."""
    out = deepcopy(base)
    for key, value in overlay.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load configs/config.yaml. "
            "Install it with: pip install pyyaml"
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at top level.")
    return data


def _env_overrides() -> dict:
    """Translate CARDIOINSIGHT_* env vars into a nested config dict."""
    mapping = {
        "CARDIOINSIGHT_RANDOM_SEED": ("project", "random_seed"),
        "CARDIOINSIGHT_DATA_PATH": ("data", "path"),
        "CARDIOINSIGHT_MODEL_ENGINE": ("model", "engine"),
        "CARDIOINSIGHT_MLFLOW_ENABLED": ("mlflow", "enabled"),
        "CARDIOINSIGHT_MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri"),
        "CARDIOINSIGHT_MLFLOW_EXPERIMENT": ("mlflow", "experiment_name"),
        "CARDIOINSIGHT_DVC_ENABLED": ("dvc", "enabled"),
    }
    nested: dict = {}
    for env_key, path in mapping.items():
        value = os.environ.get(env_key)
        if value is None:
            continue
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
        elif lowered.isdigit():
            value = int(lowered)
        node = nested
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return nested


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML config and overlay environment overrides.

    Parameters
    ----------
    path:
        Optional path to a config file. Defaults to ``configs/config.yaml``
        relative to the project root.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Create it from configs/config.yaml."
        )
    config = _load_yaml(config_path)
    config = _deep_merge(config, _env_overrides())
    return config


def get_config(path: str | Path | None = None) -> dict:
    """Return the cached config, loading it on first access.

    Pass ``path`` only on the very first call in a process; subsequent calls
    ignore it.  Call :func:`reset_config` to force a reload (useful in tests).
    """
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None or path is not None:
        _CACHED_CONFIG = load_config(path)
    return _CACHED_CONFIG


def reset_config() -> None:
    """Clear the cached config so the next :func:`get_config` reloads it."""
    global _CACHED_CONFIG
    _CACHED_CONFIG = None