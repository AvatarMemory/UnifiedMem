import os
from pathlib import Path
from typing import Optional


def _load_dotenv_layers(env_name: Optional[str] = None):
    """Load dotenv files in order, from general to specific:
    1. .env
    2. .env.<env_name> (if env_name provided)
    3. .env.local

    Existing process env always wins. Later dotenv files only fill missing values.
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


def _normalize_env_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "none":
            return None
    return value


def _get_first_env(keys: list[str], default=None):
    for key in keys:
        val = _normalize_env_value(os.getenv(key))
        if val is not None:
            return val
    return default


def _canonical_stage(stage: str) -> str:
    normalized = (stage or "").strip().lower()
    mapping = {
        "qa_gen": "qa",
        "qa_generation": "qa",
        "retrieval": "retrieve",
    }
    return mapping.get(normalized, normalized)


def _stage_prefix(stage: str) -> str:
    normalized = _canonical_stage(stage)
    mapping = {
        "index": "INDEX",
        "retrieve": "RETRIEVE",
        "qa": "QA",
        "qa_eval": "QA_EVAL",
    }
    return mapping.get(normalized, normalized.upper())


def _stage_model_keys(stage: str) -> list[str]:
    prefix = _stage_prefix(stage)
    stage_name = _canonical_stage(stage)
    keys = [f"{prefix}_LLM_MODEL", f"{prefix}_MODEL"]
    if stage_name == "qa":
        keys.append("QA_LLM")
    return keys


def _stage_base_url_keys(stage: str) -> list[str]:
    prefix = _stage_prefix(stage)
    stage_name = _canonical_stage(stage)
    keys = [f"{prefix}_BASE_URL"]
    if stage_name == "qa":
        keys.append("QA_API_BASE")
    return keys


def _stage_api_key_keys(stage: str) -> list[str]:
    prefix = _stage_prefix(stage)
    stage_name = _canonical_stage(stage)
    keys = [f"{prefix}_API_KEY"]
    if stage_name == "qa":
        keys.append("QA_API_KEY")
    return keys


def get_stage_model(stage: str, default: str = "gpt-4o-mini") -> str:
    keys = _stage_model_keys(stage)
    keys.append("LLM_MODEL")
    return _get_first_env(keys, default)


def get_stage_base_url(stage: str, default=None):
    keys = _stage_base_url_keys(stage)
    keys.extend(["OPENAI_BASE_URL", "LLM_BASE_URL"])
    return _get_first_env(keys, default)


def get_stage_api_key(stage: str, default=None):
    keys = _stage_api_key_keys(stage)
    keys.extend(["OPENAI_API_KEY", "LLM_API_KEY", "OPENAI_KEY"])
    return _get_first_env(keys, default)


def get_embedding_model(default: str = "text-embedding-3-small") -> str:
    return _get_first_env(["EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL"], default)


def get_embedding_base_url(default=None):
    return _get_first_env(
        ["EMBEDDING_API_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "LLM_BASE_URL"],
        default,
    )


def get_embedding_api_key(default=None):
    return _get_first_env(
        ["EMBEDDING_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY", "OPENAI_KEY"],
        default,
    )


def get_stage_client_config(
    stage: str,
    default_model: str = "gpt-4o-mini",
    default_base_url=None,
    default_api_key=None,
):
    return {
        "model": get_stage_model(stage, default=default_model),
        "base_url": get_stage_base_url(stage, default=default_base_url),
        "api_key": get_stage_api_key(stage, default=default_api_key),
    }


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


__all__ = [
    "getenv",
    "get_int",
    "get_float",
    "get_bool",
    "get_all_with_prefix",
    "get_stage_model",
    "get_stage_base_url",
    "get_stage_api_key",
    "get_stage_client_config",
    "get_embedding_model",
    "get_embedding_base_url",
    "get_embedding_api_key",
    "ENV_NAME",
]
