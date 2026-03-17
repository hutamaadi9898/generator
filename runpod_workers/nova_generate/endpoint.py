from __future__ import annotations

from typing import Any

from runpod_workers.common import (
    endpoint_env,
    shared_volume,
)
from runpod_workers.flash_compat import load_endpoint_and_gpu_group
from runpod_workers.nova_generate.core import nova_generate as _nova_generate

Endpoint, GpuGroup = load_endpoint_and_gpu_group()

@Endpoint(
    name="nova-generate",
    gpu=[
        GpuGroup.ADA_24,
        GpuGroup.AMPERE_24,
        GpuGroup.AMPERE_48,
    ],
    workers=(0, 1),
    idle_timeout=300,
    flashboot=True,
    dependencies=[
        "torch>=2.5.1",
        "diffusers>=0.35.0",
        "transformers>=4.48.0",
        "accelerate>=1.2.1",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.29.0",
        "Pillow>=11.0.0",
    ],
    system_dependencies=[
        "libgl1",
        "libglib2.0-0",
    ],
    volume=shared_volume(),
    env=endpoint_env(),
)
async def nova_generate(**payload: Any) -> dict[str, Any]:
    return await _nova_generate(**payload)
