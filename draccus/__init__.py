"""Compatibility shim for draccus.

If the real `draccus` package is installed, this module simply proxies to it.
Otherwise, we provide a lightweight stub that implements the APIs consumed by
Levanter's tests and configuration helpers so the project remains runnable in
restricted environments.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]

# Temporarily remove the project root from sys.path so we can detect an actual
# installation of draccus (e.g., from site-packages) rather than re-discovering
# this shim.
_removed = False
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
    _removed = True

try:
    _spec = importlib.util.find_spec("draccus")
finally:
    if _removed:
        sys.path.insert(0, str(_PROJECT_ROOT))

if _spec is not None and _spec.origin is not None and Path(_spec.origin).resolve() != _THIS_FILE:
    # Delegate to the real draccus implementation.
    module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = module
    assert _spec.loader is not None
    _spec.loader.exec_module(module)
else:
    # ---------------------------------------------------------------------
    # Minimal fallback implementation used when draccus isn't available.
    # ---------------------------------------------------------------------
    import dataclasses
    import inspect
    import os
    from dataclasses import field as dataclasses_field
    from typing import Any, Dict, Iterable, Tuple, TypeVar, Union, get_args, get_origin

    import yaml

    __all__ = [
        "ChoiceRegistry",
        "PluginRegistry",
        "decode",
        "encode",
        "field",
        "parse",
    ]

    T = TypeVar("T")

    def field(*args, **kwargs):  # type: ignore[misc]
        """Alias to :func:`dataclasses.field` to match the draccus API."""

        return dataclasses_field(*args, **kwargs)

    class ChoiceRegistry:
        """Lightweight choice registry mirroring draccus helpers."""

        _registry: Dict[type, Dict[str, type]] = {}
        discover_packages_path: str | None = None

        def __init_subclass__(cls, choice_name: str | None = None, **kwargs):
            cls.discover_packages_path = kwargs.pop(
                "discover_packages_path", getattr(cls, "discover_packages_path", None)
            )
            super().__init_subclass__(**kwargs)
            if choice_name is not None:
                cls.register_subclass(choice_name)(cls)

        @classmethod
        def _choices(cls) -> Dict[str, type]:
            return ChoiceRegistry._registry.setdefault(cls, {})

        @classmethod
        def register_subclass(cls, choice_name: str, subclass: type | None = None):
            def decorator(inner_cls: type) -> type:
                cls._choices()[choice_name] = inner_cls
                setattr(inner_cls, "_choice_name", choice_name)
                return inner_cls

            if subclass is not None:
                return decorator(subclass)
            return decorator

        @classmethod
        def choices(cls) -> Dict[str, type]:
            return dict(cls._choices())

        @classmethod
        def default_choice_name(cls) -> str | None:
            return getattr(cls, "_default_choice_name", None)

        @classmethod
        def get_choice_class(cls, choice_name: str | None):
            if choice_name is None:
                choice_name = cls.default_choice_name()
            if choice_name is None:
                raise KeyError("No default choice registered")
            choices = cls._choices()
            if choice_name not in choices:
                raise KeyError(choice_name)
            return choices[choice_name]

    class PluginRegistry(ChoiceRegistry):
        pass

    _decode_hooks: Dict[type, Any] = {}
    _encode_hooks: Dict[type, Any] = {}

    def _is_dataclass_type(tp: Any) -> bool:
        return inspect.isclass(tp) and dataclasses.is_dataclass(tp)

    def _strip_optional(tp: Any) -> Tuple[Any, bool]:
        origin = get_origin(tp)
        if origin is None:
            return tp, False
        if origin in (tuple, list, dict, Iterable):
            return tp, False
        if origin is Union:
            args = [arg for arg in get_args(tp) if arg is not type(None)]  # noqa: E721
            if len(args) == 1:
                return args[0], True
        return tp, False

    def _decode_dataclass(cls: type, data: Dict[str, Any] | None):
        if data is None:
            data = {}
        kwargs = {}
        for field in dataclasses.fields(cls):
            field_value = data.get(field.name, dataclasses.MISSING)
            if field_value is dataclasses.MISSING:
                continue
            field_type = field.type
            decoded = _decode_inner(field_type, field_value)
            kwargs[field.name] = decoded
        return cls(**kwargs)

    def _decode_inner(tp: Any, data: Any):
        hook = _decode_hooks.get(tp)
        if hook is not None:
            return hook(data)

        if data is None:
            base_type, was_optional = _strip_optional(tp)
            if was_optional:
                return None
            tp = base_type

        origin = get_origin(tp)

        if origin is tuple and isinstance(data, list):
            args = get_args(tp)
            if len(args) == 2 and args[1] is ...:
                return tuple(_decode_inner(args[0], item) for item in data)
            return tuple(_decode_inner(arg, item) for arg, item in zip(args, data))

        if origin in (list, Iterable) and isinstance(data, list):
            (inner,) = get_args(tp)
            return [_decode_inner(inner, item) for item in data]

        if origin is dict and isinstance(data, dict):
            key_type, val_type = get_args(tp)
            return {
                _decode_inner(key_type, k): _decode_inner(val_type, v) for k, v in data.items()
            }

        if inspect.isclass(tp) and issubclass(tp, ChoiceRegistry):
            if isinstance(data, dict):
                choice_name = data.get("type") or data.get("name")
                config_cls = tp.get_choice_class(choice_name)
                inner = {k: v for k, v in data.items() if k not in {"type", "name"}}
                return _decode_inner(config_cls, inner)
            if isinstance(data, str):
                return tp.get_choice_class(data)()
            config_cls = tp.get_choice_class(None)
            return _decode_inner(config_cls, data)

        if _is_dataclass_type(tp):
            if isinstance(data, dict):
                return _decode_dataclass(tp, data)
            if data is None:
                return tp()

        return data

    class _DecodeDispatcher:
        def register(self, tp: type, fn):
            _decode_hooks[tp] = fn

        def __call__(self, tp, data):
            return _decode_inner(tp, data)

    class _EncodeDispatcher:
        def register(self, tp: type, fn):
            _encode_hooks[tp] = fn

        def __call__(self, obj):
            hook = _encode_hooks.get(type(obj))
            if hook is not None:
                return hook(obj)
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            return obj

    decode = _DecodeDispatcher()
    encode = _EncodeDispatcher()

    def parse(config_class: type[T], config_file: str, args: Iterable[str] | None = None) -> T:
        with open(os.path.expanduser(config_file), "r", encoding="utf-8") as f:
            contents = yaml.safe_load(f) or {}
        return decode(config_class, contents)
