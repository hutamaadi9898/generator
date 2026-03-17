from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from runpod_workers.common import (
    encode_file_base64,
    ensure_dir,
    ensure_volume_path,
    env_bool,
    flatten_caption_tags,
    hash_text,
    resolve_model_source,
    slugify,
    timestamp_slug,
    write_json,
)


_PIPELINES: dict[str, Any] = {}
_ACTIVE_LORAS: dict[int, str] = {}


async def nova_generate(**payload: Any) -> dict[str, Any]:
    return await asyncio.to_thread(generate_sync, dict(payload))


def generate_sync(payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler
    from PIL import Image

    prompt = str(payload.get("prompt") or "").strip()
    negative_prompt = str(payload.get("negative_prompt") or "").strip()
    base_model_ref = str(payload.get("base_model_ref") or "").strip()
    output_dir = str(payload.get("output_dir") or "").strip()
    if not prompt:
        raise ValueError("`prompt` is required.")
    if not base_model_ref:
        raise ValueError("`base_model_ref` is required.")
    if not output_dir:
        raise ValueError("`output_dir` is required.")

    width = int(payload.get("width") or 1024)
    height = int(payload.get("height") or 1024)
    steps = int(payload.get("steps") or 30)
    cfg = float(payload.get("cfg") or 6.0)
    seed = int(payload.get("seed") or 0)
    mode = str(payload.get("mode") or "dataset").strip() or "dataset"
    lora_path = str(payload.get("lora_path") or "").strip()
    lora_weight = float(payload.get("lora_weight") or 0.8)
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw": metadata}

    pipeline = _load_pipeline(AutoPipelineForText2Image, EulerAncestralDiscreteScheduler, torch, base_model_ref)
    _apply_lora(pipeline, lora_path, lora_weight)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    )
    image: Image.Image = result.images[0]

    output_root = ensure_volume_path(output_dir, create=True)
    shot_label = slugify(str(metadata.get("shot_label") or mode), mode)
    filename = f"{shot_label}-{seed}-{hash_text(prompt)}.png"
    image_path = output_root / filename
    image.save(image_path, format="PNG")

    metadata_payload = {
        "mode": mode,
        "base_model_ref": base_model_ref,
        "resolved_model_source": resolve_model_source(base_model_ref),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "lora_path": lora_path,
        "lora_weight": lora_weight,
        "metadata": metadata,
        "caption_tags": flatten_caption_tags(str(metadata.get("caption_tags") or "")),
        "saved_at": timestamp_slug(),
    }
    write_json(image_path.with_suffix(".json"), metadata_payload)

    response = {
        "image_path": str(image_path),
        "metadata": metadata_payload,
    }
    if env_bool("NOVA_RETURN_IMAGE_BASE64", True):
        response["image_base64"] = encode_file_base64(image_path)
    return response


def _load_pipeline(AutoPipelineForText2Image, EulerAncestralDiscreteScheduler, torch, base_model_ref: str):
    model_source = resolve_model_source(base_model_ref)
    if model_source in _PIPELINES:
        return _PIPELINES[model_source]

    dtype_name = os.getenv("NOVA_TORCH_DTYPE", "float16").strip().lower()
    dtype = getattr(torch, dtype_name, torch.float16)
    model_cache = ensure_dir(Path(os.getenv("NOVA_MODEL_CACHE_DIR", "/runpod-volume/models")))
    source_path = Path(model_source)
    kwargs = {
        "torch_dtype": dtype,
        "token": os.getenv("HF_TOKEN") or None,
        "use_safetensors": env_bool("NOVA_USE_SAFETENSORS", True),
    }

    if source_path.exists() and source_path.is_file():
        pipeline = AutoPipelineForText2Image.from_single_file(str(source_path), **kwargs)
    elif source_path.exists():
        pipeline = AutoPipelineForText2Image.from_pretrained(str(source_path), **kwargs)
    else:
        pipeline = AutoPipelineForText2Image.from_pretrained(model_source, cache_dir=str(model_cache), **kwargs)

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipeline.enable_vae_tiling()
    except Exception:
        pass
    try:
        pipeline.enable_attention_slicing()
    except Exception:
        pass

    pipeline.to("cuda")
    _PIPELINES[model_source] = pipeline
    return pipeline


def _apply_lora(pipeline, lora_path: str, lora_weight: float) -> None:
    pipeline_id = id(pipeline)
    active = _ACTIVE_LORAS.get(pipeline_id, "")
    if not lora_path:
        if active:
            _unload_lora(pipeline, pipeline_id)
        return

    resolved_path = ensure_volume_path(lora_path, file_ok=True)
    if active != str(resolved_path):
        _unload_lora(pipeline, pipeline_id)
        pipeline.load_lora_weights(str(resolved_path), adapter_name="request")
        _ACTIVE_LORAS[pipeline_id] = str(resolved_path)

    if hasattr(pipeline, "set_adapters"):
        pipeline.set_adapters(["request"], adapter_weights=[lora_weight])
    elif hasattr(pipeline, "fuse_lora"):
        pipeline.fuse_lora(lora_scale=lora_weight)
    else:
        raise RuntimeError("Loaded pipeline does not support LoRA adapters.")


def _unload_lora(pipeline, pipeline_id: int) -> None:
    if hasattr(pipeline, "unload_lora_weights"):
        try:
            pipeline.unload_lora_weights()
        except Exception:
            pass
    if hasattr(pipeline, "delete_adapters"):
        try:
            pipeline.delete_adapters(["request"])
        except Exception:
            pass
    _ACTIVE_LORAS.pop(pipeline_id, None)


__all__ = ["generate_sync", "nova_generate"]
