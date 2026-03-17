from __future__ import annotations

import importlib
from typing import Any


def _get_attr(module: Any, module_name: str, attr: str) -> Any:
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"{module_name} does not export {attr}") from exc


def _import_attr(module_name: str, attr: str) -> Any:
    return _get_attr(importlib.import_module(module_name), module_name, attr)


def load_endpoint_and_gpu_group() -> tuple[Any, Any]:
    flash = importlib.import_module("runpod_flash")

    try:
        endpoint = _get_attr(flash, "runpod_flash", "Endpoint")
    except ImportError:
        endpoint = _import_attr("runpod_flash.endpoint", "Endpoint")

    try:
        gpu_group = _get_attr(flash, "runpod_flash", "GpuGroup")
    except ImportError:
        gpu_group = _import_attr("runpod_flash.core.resources", "GpuGroup")

    return endpoint, gpu_group


__all__ = ["load_endpoint_and_gpu_group"]
