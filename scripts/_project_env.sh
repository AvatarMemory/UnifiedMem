#!/bin/bash

_call_config_helper() {
    local helper_name="$1"
    local arg="$2"
    local default_value="${3-}"
    local repo_root="$4"
    python - "$helper_name" "$arg" "$default_value" "$repo_root" <<'PY'
import sys

helper_name = sys.argv[1]
arg = sys.argv[2]
default = sys.argv[3]
repo_root = sys.argv[4]

sys.path.insert(0, repo_root)
from src import config as cfg

helper = getattr(cfg, helper_name)
if helper_name == "getenv":
    value = helper(arg, default)
else:
    value = helper(arg, default=default)

print("" if value is None else value)
PY
}

get_project_env() {
    _call_config_helper "getenv" "$1" "${2-}" "$3"
}

get_stage_model() {
    _call_config_helper "get_stage_model" "$1" "${2-}" "$3"
}

get_stage_base_url() {
    _call_config_helper "get_stage_base_url" "$1" "${2-}" "$3"
}

get_stage_api_key() {
    _call_config_helper "get_stage_api_key" "$1" "${2-}" "$3"
}
