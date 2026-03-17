from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runpod_workers.flash_compat as flash_compat


def test_load_endpoint_and_gpu_group_prefers_top_level_exports(monkeypatch) -> None:
    endpoint = object()
    gpu_group = object()

    def fake_import(name: str):
        if name != "runpod_flash":
            raise AssertionError(f"unexpected module import: {name}")
        return SimpleNamespace(Endpoint=endpoint, GpuGroup=gpu_group)

    monkeypatch.setattr(flash_compat.importlib, "import_module", fake_import)

    resolved_endpoint, resolved_gpu_group = flash_compat.load_endpoint_and_gpu_group()

    assert resolved_endpoint is endpoint
    assert resolved_gpu_group is gpu_group


def test_load_endpoint_and_gpu_group_avoids_endpoint_fallback_when_only_gpu_group_is_missing(monkeypatch) -> None:
    endpoint = object()
    gpu_group = object()
    calls: list[str] = []

    def fake_import(name: str):
        calls.append(name)
        if name == "runpod_flash":
            return SimpleNamespace(Endpoint=endpoint)
        if name == "runpod_flash.core.resources":
            return SimpleNamespace(GpuGroup=gpu_group)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(flash_compat.importlib, "import_module", fake_import)

    resolved_endpoint, resolved_gpu_group = flash_compat.load_endpoint_and_gpu_group()

    assert resolved_endpoint is endpoint
    assert resolved_gpu_group is gpu_group
    assert "runpod_flash.endpoint" not in calls


def test_load_endpoint_and_gpu_group_falls_back_for_endpoint_only(monkeypatch) -> None:
    endpoint = object()
    gpu_group = object()

    def fake_import(name: str):
        if name == "runpod_flash":
            return SimpleNamespace(GpuGroup=gpu_group)
        if name == "runpod_flash.endpoint":
            return SimpleNamespace(Endpoint=endpoint)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(flash_compat.importlib, "import_module", fake_import)

    resolved_endpoint, resolved_gpu_group = flash_compat.load_endpoint_and_gpu_group()

    assert resolved_endpoint is endpoint
    assert resolved_gpu_group is gpu_group
