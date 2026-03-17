from __future__ import annotations

import asyncio
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


def test_handler_executes_async_worker_with_running_event_loop(monkeypatch) -> None:
    async def fake_generate(**payload):
        await asyncio.sleep(0)
        return {"ok": payload["prompt"]}

    monkeypatch.setattr(serverless_main, "_HANDLERS", {"nova_generate": fake_generate})
    monkeypatch.setenv("RUNPOD_HANDLER", "nova_generate")

    async def run_test():
        return await asyncio.to_thread(serverless_main.handler, {"input": {"prompt": "loop"}})

    assert asyncio.run(run_test()) == {"ok": "loop"}
