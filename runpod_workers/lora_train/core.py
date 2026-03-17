from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import Any

from runpod_workers.common import (
    ensure_dir,
    ensure_volume_path,
    env_bool,
    env_float,
    env_int,
    flatten_caption_tags,
    link_or_copy,
    newest_matching_file,
    prepare_local_model_source,
    run_logged_command,
    slugify,
    write_json,
)


async def lora_train(**payload: Any) -> dict[str, Any]:
    return await asyncio.to_thread(train_sync, dict(payload))


def train_sync(payload: dict[str, Any]) -> dict[str, Any]:
    base_model_ref = str(payload.get("base_model_ref") or "").strip()
    output_path = str(payload.get("output_path") or "").strip()
    trigger_token = str(payload.get("trigger_token") or "").strip()
    selected_images = payload.get("selected_images") or []
    config = payload.get("config") or {}
    preset_name = str(payload.get("preset_name") or "").strip()

    if not base_model_ref:
        raise ValueError("`base_model_ref` is required.")
    if not output_path:
        raise ValueError("`output_path` is required.")
    if not trigger_token:
        raise ValueError("`trigger_token` is required.")
    if not isinstance(selected_images, list) or not selected_images:
        raise ValueError("`selected_images` must contain at least one curated dataset image.")

    output_root = ensure_volume_path(output_path, create=True)
    logs_dir = ensure_dir(output_root / "logs")
    previews_dir = ensure_dir(output_root / "previews")
    dataset_root = ensure_dir(output_root / "dataset")
    train_dir = ensure_dir(dataset_root / f"{_compute_repeats(config, len(selected_images))}_{slugify(trigger_token, 'token')}")
    log_path = logs_dir / "train.log"

    manifest = _build_dataset_manifest(selected_images, trigger_token, train_dir)
    write_json(output_root / "request.json", payload)
    write_json(output_root / "dataset_manifest.json", manifest)

    model_cache = ensure_dir(Path(os.getenv("NOVA_MODEL_CACHE_DIR", "/runpod-volume/models")))
    model_source = prepare_local_model_source(base_model_ref, model_cache)
    sd_scripts_dir = _ensure_sd_scripts_repo(log_path)
    sample_prompts = _write_sample_prompts(previews_dir, trigger_token)
    command = _build_training_command(
        sd_scripts_dir=sd_scripts_dir,
        model_source=model_source,
        train_dir=train_dir,
        output_root=output_root,
        logs_dir=logs_dir,
        sample_prompts=sample_prompts,
        config=config,
    )

    run_logged_command(command, log_path=log_path, cwd=sd_scripts_dir)

    lora_path = newest_matching_file(output_root, "*.safetensors")
    if lora_path is None:
        lora_path = newest_matching_file(output_root, "**/*.safetensors")
    if lora_path is None:
        raise RuntimeError(f"Training finished but no `.safetensors` weights were found under {output_root}.")

    sample_previews = sorted(str(path) for path in previews_dir.glob("*.png"))
    result = {
        "preset_name": preset_name,
        "lora_path": str(lora_path),
        "logs_path": str(log_path),
        "sample_previews": sample_previews,
        "dataset_manifest_path": str(output_root / "dataset_manifest.json"),
    }
    write_json(output_root / "result.json", result)
    return result


def _build_dataset_manifest(selected_images: list[dict[str, Any]], trigger_token: str, train_dir: Path) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for index, image in enumerate(selected_images, start=1):
        remote_image_path = str(image.get("remote_image_path") or "").strip()
        if not remote_image_path:
            raise ValueError(f"selected_images[{index - 1}] is missing `remote_image_path`.")
        source = ensure_volume_path(remote_image_path, file_ok=True)
        if not source.exists():
            raise FileNotFoundError(f"Curated dataset image does not exist on the volume: {source}")

        caption = flatten_caption_tags(image.get("caption_tags") or [])
        if trigger_token not in caption:
            caption = flatten_caption_tags([trigger_token, caption])

        target_image = train_dir / f"{index:04d}{source.suffix.lower()}"
        target_caption = target_image.with_suffix(".txt")
        link_or_copy(source, target_image)
        target_caption.write_text(caption, encoding="utf-8")
        manifest.append(
            {
                "source_image": str(source),
                "train_image": str(target_image),
                "caption_path": str(target_caption),
                "caption_text": caption,
            }
        )
    return manifest


def _compute_repeats(config: dict[str, Any], image_count: int) -> int:
    if image_count <= 0:
        return 1
    if config.get("dataset_repeats"):
        return max(1, int(config["dataset_repeats"]))
    max_train_steps = int(config.get("max_train_steps") or 1200)
    batch_size = int(config.get("batch_size") or 1)
    approx_epoch_images = max(1, math.ceil(max_train_steps * batch_size / max(image_count, 1)))
    return max(1, min(20, approx_epoch_images))


def _ensure_sd_scripts_repo(log_path: Path) -> Path:
    repo_url = os.getenv("SD_SCRIPTS_REPO", "https://github.com/kohya-ss/sd-scripts.git").strip()
    repo_ref = os.getenv("SD_SCRIPTS_REF", "main").strip() or "main"
    repo_root = ensure_dir(Path(os.getenv("SD_SCRIPTS_DIR", "/runpod-volume/tooling")) / "sd-scripts")

    if not (repo_root / ".git").exists():
        run_logged_command(["git", "clone", repo_url, str(repo_root)], log_path=log_path)
    if env_bool("SD_SCRIPTS_AUTO_UPDATE", True):
        run_logged_command(["git", "fetch", "--all", "--tags"], log_path=log_path, cwd=repo_root)
        run_logged_command(["git", "checkout", repo_ref], log_path=log_path, cwd=repo_root)
        run_logged_command(["git", "pull", "--ff-only"], log_path=log_path, cwd=repo_root)
    return repo_root


def _write_sample_prompts(previews_dir: Path, trigger_token: str) -> Path:
    prompts_path = previews_dir / "sample_prompts.txt"
    prompts = [
        f"{trigger_token}, anime illustration, portrait, soft smile, highly detailed",
        f"{trigger_token}, anime illustration, half body, school uniform, cinematic lighting",
        f"{trigger_token}, anime illustration, full body, dynamic pose, clean line art",
    ]
    prompts_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    return prompts_path


def _build_training_command(
    *,
    sd_scripts_dir: Path,
    model_source: Path,
    train_dir: Path,
    output_root: Path,
    logs_dir: Path,
    sample_prompts: Path,
    config: dict[str, Any],
) -> list[str]:
    resolution = str(config.get("resolution") or "1024x1024").lower().replace("x", ",")
    rank = int(config.get("rank") or 16)
    alpha = int(config.get("alpha") or rank)
    batch_size = int(config.get("batch_size") or 1)
    grad_accum = int(config.get("grad_accum") or 4)
    precision = str(config.get("precision") or "bf16").strip().lower()
    max_train_steps = int(config.get("max_train_steps") or 1200)
    sample_every = int(config.get("sample_every") or 200)
    learning_rate = env_float("LORA_LEARNING_RATE", 1e-4)
    text_encoder_lr = env_float("LORA_TEXT_ENCODER_LR", 5e-5)
    unet_lr = env_float("LORA_UNET_LR", 1e-4)
    optimizer = str(config.get("optimizer") or "AdamW8bit").strip()
    output_name = os.getenv("LORA_OUTPUT_NAME", "final").strip() or "final"
    save_precision = os.getenv("LORA_SAVE_PRECISION", precision).strip().lower()

    command = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process",
        str(env_int("LORA_CPU_THREADS", 2)),
        str(sd_scripts_dir / "sdxl_train_network.py"),
        "--pretrained_model_name_or_path",
        str(model_source),
        "--train_data_dir",
        str(train_dir.parent),
        "--output_dir",
        str(output_root),
        "--logging_dir",
        str(logs_dir),
        "--resolution",
        resolution,
        "--output_name",
        output_name,
        "--network_module",
        "networks.lora",
        "--network_dim",
        str(rank),
        "--network_alpha",
        str(alpha),
        "--optimizer_type",
        optimizer,
        "--learning_rate",
        str(learning_rate),
        "--text_encoder_lr",
        str(text_encoder_lr),
        "--unet_lr",
        str(unet_lr),
        "--train_batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--max_train_steps",
        str(max_train_steps),
        "--save_model_as",
        "safetensors",
        "--mixed_precision",
        precision,
        "--save_precision",
        save_precision,
        "--caption_extension",
        ".txt",
        "--cache_latents",
        "--cache_latents_to_disk",
        "--gradient_checkpointing",
        "--max_data_loader_n_workers",
        "0",
        "--seed",
        str(env_int("LORA_SEED", 42)),
    ]

    if sample_every > 0:
        command.extend(
            [
                "--save_every_n_steps",
                str(sample_every),
                "--sample_every_n_steps",
                str(sample_every),
                "--sample_prompts",
                str(sample_prompts),
            ]
        )
    if env_bool("LORA_USE_SDPA", True):
        command.append("--sdpa")
    if env_bool("LORA_USE_XFORMERS", False):
        command.append("--xformers")
    if env_bool("LORA_ENABLE_BUCKETS", True):
        command.append("--enable_bucket")
    if env_bool("LORA_SAVE_STATE", False):
        command.append("--save_state")
    return command


__all__ = ["lora_train", "train_sync"]
