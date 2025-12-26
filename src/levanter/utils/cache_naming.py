import dataclasses
import hashlib
import json
import os
from typing import Any, Iterable, Mapping


def _to_primitive(value: Any, *, exclude_keys: Iterable[str]) -> Any:
    """Convert nested dataclasses/containers to JSON-serializable primitives.

    Non-serializable leaf values fall back to repr for stability.
    Excludes keys provided in exclude_keys when traversing dataclasses and dicts.
    """
    if dataclasses.is_dataclass(value):
        result: dict[str, Any] = {}
        for field in dataclasses.fields(value):
            name = field.name
            if name in exclude_keys:
                continue
            result[name] = _to_primitive(getattr(value, name), exclude_keys=exclude_keys)
        return result

    if isinstance(value, Mapping):
        # Sort keys for deterministic order
        return {str(k): _to_primitive(v, exclude_keys=exclude_keys) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}

    if isinstance(value, (list, tuple)):
        return [_to_primitive(v, exclude_keys=exclude_keys) for v in value]

    if isinstance(value, set):
        return [_to_primitive(v, exclude_keys=exclude_keys) for v in sorted(value, key=lambda x: repr(x))]

    # Try JSON encode directly; otherwise use repr as a stable string
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def hash_for_cache_config(config_obj: Any, *, extra: Mapping[str, Any] | None = None, exclude_keys: Iterable[str] = ("cache_dir",)) -> str:
    """Compute a short, deterministic hash representing the cache-relevant parts of a config.

    - Traverses dataclasses and containers into primitives
    - Excludes keys like cache_dir that shouldn't affect the cache content
    - Allows injecting extra fields (e.g., tokenizer id) to the hash basis
    """
    prim = _to_primitive(config_obj, exclude_keys=exclude_keys)
    if extra:
        # Merge extra after conversion, under a reserved key to avoid collisions
        prim = {"__config__": prim, "__extra__": _to_primitive(extra, exclude_keys=exclude_keys)}

    payload = json.dumps(prim, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def hashed_cache_dir(base_cache_dir: str, config_obj: Any, *, extra: Mapping[str, Any] | None = None, exclude_keys: Iterable[str] = ("cache_dir",)) -> str:
    """Append a stable short hash subdirectory under base_cache_dir for the given config."""
    h = hash_for_cache_config(config_obj, extra=extra, exclude_keys=exclude_keys)
    return os.path.join(base_cache_dir, h)


