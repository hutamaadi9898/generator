from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import os
from typing import Any

from runpod_workers.lora_train.core import lora_train
from runpod_workers.nova_generate.core import nova_generate


LOGGER = logging.getLogger(__name__)
_HANDLERS = {
    "lora_train": lora_train,
    "nova_generate": nova_generate,
}


def _resolve_handler_name() -> str:
    handler_name = os.getenv("RUNPOD_HANDLER", "nova_generate").strip()
    if handler_name not in _HANDLERS:
        available = ", ".join(sorted(_HANDLERS))
        raise RuntimeError(f"Unsupported RUNPOD_HANDLER '{handler_name}'. Expected one of: {available}")
    return handler_name


def _run_maybe_async(result: Any) -> Any:
    if inspect.isawaitable(result):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, result).result()
    return result


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input", {})
    if not isinstance(job_input, dict):
        raise ValueError("RunPod input must be a JSON object.")

    handler_name = _resolve_handler_name()
    LOGGER.info("Executing handler %s", handler_name)
    return _run_maybe_async(_HANDLERS[handler_name](**job_input))


if __name__ == "__main__":
    import runpod

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    runpod.serverless.start({"handler": handler})
