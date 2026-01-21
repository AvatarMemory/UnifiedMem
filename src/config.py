import os
from pathlib import Path
from typing import Optional


def _load_dotenv_layers(env_name: Optional[str] = None):
    """Load dotenv files in order, from general to specific:
    1. .env
    2. .env.<env_name> (if env_name provided)
    3. .env.local

    Later files override earlier values.
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [repo_root / ".env"]
    if env_name:
        candidates.append(repo_root / f".env.{env_name}")
    # allow a local override
    candidates.append(repo_root / ".env.local")

    for p in candidates:
        if p.exists():
            try:
                load_dotenv(dotenv_path=str(p), override=False)
            except Exception:
                # best-effort; don't fail startup
                pass


# Determine environment name from common env vars if set
ENV_NAME = os.getenv("PROJECT_ENV") or os.getenv("ENV") or os.getenv("PY_ENV")
_load_dotenv_layers(ENV_NAME)


def getenv(key: str, default=None, cast: Optional[callable] = None):
    """Get environment variable with optional casting.

    Reads from `os.environ`, which may have been populated by layered .env files above.
    """
    val = os.getenv(key, default)
    if val is None:
        return default
    if cast is not None:
        try:
            return cast(val)
        except Exception:
            return default
    return val


def get_int(key: str, default: int = 0) -> int:
    return getenv(key, default=default, cast=int)


def get_float(key: str, default: float = 0.0) -> float:
    return getenv(key, default=default, cast=float)


def get_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def get_all_with_prefix(prefix: str):
    """Return a dict of environment variables that start with `prefix`."""
    return {k: v for k, v in os.environ.items() if k.startswith(prefix)}


__all__ = ["getenv", "get_int", "get_float", "get_bool", "get_all_with_prefix", "ENV_NAME"]
