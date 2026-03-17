from __future__ import annotations

from typing import Any

from runpod_workers.common import (
    endpoint_env,
    shared_volume,
)
from runpod_workers.flash_compat import load_endpoint_and_gpu_group
from runpod_workers.lora_train.core import lora_train as _lora_train

Endpoint, GpuGroup = load_endpoint_and_gpu_group()


@Endpoint(
    name="lora-train",
    gpu=[
        GpuGroup.AMPERE_48,
        GpuGroup.ADA_48_PRO,
        GpuGroup.AMPERE_80,
    ],
    workers=(0, 1),
    idle_timeout=900,
    flashboot=True,
    execution_timeout_ms=0,
    dependencies=[
        "torch>=2.5.1",
        "accelerate>=1.2.1",
        "diffusers>=0.35.0",
        "transformers>=4.48.0",
        "huggingface_hub>=0.29.0",
        "safetensors>=0.4.5",
        "bitsandbytes>=0.45.0; python_version < '3.14'",
        "sentencepiece>=0.2.0",
        "Pillow>=11.0.0",
    ],
    system_dependencies=[
        "git",
        "libgl1",
        "libglib2.0-0",
    ],
    volume=shared_volume(),
    env=endpoint_env(),
)
async def lora_train(**payload: Any) -> dict[str, Any]:
    return await _lora_train(**payload)
