from __future__ import annotations

import pytest

import runpod_serverless.main as serverless_main


def test_handler_dispatches_selected_worker(monkeypatch) -> None:
    async def fake_generate(**payload):
        return {"ok": payload["prompt"]}

    monkeypatch.setattr(serverless_main, "_HANDLERS", {"nova_generate": fake_generate})
    monkeypatch.setenv("RUNPOD_HANDLER", "nova_generate")

    assert serverless_main.handler({"input": {"prompt": "test"}}) == {"ok": "test"}


def test_handler_rejects_non_object_input(monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_HANDLER", "nova_generate")
    with pytest.raises(ValueError):
        serverless_main.handler({"input": "bad"})


def test_handler_rejects_unknown_handler(monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_HANDLER", "missing")
    with pytest.raises(RuntimeError):
        serverless_main.handler({"input": {}})
