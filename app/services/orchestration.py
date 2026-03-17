from __future__ import annotations

import base64

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import CharacterProfile, DatasetImage, GenerationJob, TrainingRun
from app.schemas import DatasetGenerateRequest, DatasetImageUpdate, PromptPack, RenderRequest
from app.services.dataset import TRAINING_PRESET, build_dataset_prompts, dataset_version
from app.services.utils import dumps_json, ensure_dir, loads_json, normalize_tags, stable_seed
from app.settings import Settings, get_settings


def list_characters(db: Session) -> list[CharacterProfile]:
    stmt = (
        select(CharacterProfile)
        .options(
            selectinload(CharacterProfile.generation_jobs),
            selectinload(CharacterProfile.dataset_images),
            selectinload(CharacterProfile.training_runs),
        )
        .order_by(CharacterProfile.created_at.desc())
    )
    return list(db.scalars(stmt).unique())


def get_character(db: Session, character_id: int) -> CharacterProfile | None:
    stmt = (
        select(CharacterProfile)
        .where(CharacterProfile.id == character_id)
        .options(
            selectinload(CharacterProfile.generation_jobs),
            selectinload(CharacterProfile.dataset_images),
            selectinload(CharacterProfile.training_runs),
        )
    )
    return db.scalars(stmt).unique().first()


def create_character(
    db: Session,
    *,
    name: str,
    trigger_token: str,
    visual_traits: str,
    default_outfit: str,
    base_model_ref: str,
) -> CharacterProfile:
    character = CharacterProfile(
        name=name.strip(),
        trigger_token=trigger_token.strip(),
        visual_traits=visual_traits.strip(),
        default_outfit=default_outfit.strip(),
        base_model_ref=base_model_ref,
    )
    db.add(character)
    db.commit()
    db.refresh(character)
    return character


def apply_prompt_pack(db: Session, character: CharacterProfile, prompt_pack: PromptPack) -> CharacterProfile:
    character.canonical_prompt = prompt_pack.canonical_prompt
    character.negative_prompt = prompt_pack.negative_prompt
    character.style_tags_json = dumps_json(normalize_tags(prompt_pack.style_tags))
    character.shot_list_json = dumps_json(prompt_pack.shot_list)
    db.add(character)
    db.commit()
    db.refresh(character)
    return character


def submit_dataset_generation(
    db: Session,
    *,
    character: CharacterProfile,
    request: DatasetGenerateRequest,
    runpod_client,
    settings: Settings,
) -> list[GenerationJob]:
    if not character.canonical_prompt:
        raise ValueError("Generate a prompt pack before submitting the dataset batch.")
    prompt_pack = PromptPack(
        canonical_prompt=character.canonical_prompt,
        negative_prompt=character.negative_prompt,
        style_tags=loads_json(character.style_tags_json, []),
        shot_list=loads_json(character.shot_list_json, []),
    )
    jobs: list[GenerationJob] = []
    for offset, item in enumerate(build_dataset_prompts(prompt_pack, character.trigger_token, character.default_outfit)):
        payload = {
            "mode": "dataset",
            "character_id": character.id,
            "base_model_ref": character.base_model_ref,
            "output_dir": f"{settings.runpod_volume_root}/characters/{character.id}/dataset",
            "prompt": item["prompt"],
            "negative_prompt": character.negative_prompt,
            "seed": stable_seed(request.seed_base, offset),
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "cfg": request.cfg,
            "metadata": {"shot_label": item["label"], "caption_tags": item["caption_tags"]},
        }
        submission = runpod_client.submit_job(settings.runpod_generate_endpoint_id, payload)
        job = GenerationJob(
            character_id=character.id,
            kind="dataset",
            source_shot=item["label"],
            prompt=item["prompt"],
            negative_prompt=character.negative_prompt,
            seed=payload["seed"],
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg=request.cfg,
            runpod_job_id=submission.job_id,
            status=submission.status,
            output_json=dumps_json(
                {
                    "artifacts": submission.artifacts,
                    "caption_tags": item["caption_tags"],
                }
            ),
        )
        db.add(job)
        jobs.append(job)
    db.commit()
    for job in jobs:
        db.refresh(job)
    return jobs


def sync_generation_jobs(db: Session, character: CharacterProfile, runpod_client, settings: Settings) -> None:
    pending = [job for job in character.generation_jobs if job.status not in {"COMPLETED", "FAILED", "CANCELLED"}]
    for job in pending:
        state = runpod_client.get_status(settings.runpod_generate_endpoint_id, job.runpod_job_id)
        job.status = state.status
        output_json = loads_json(job.output_json, {})
        output_json["artifacts"] = state.output
        preview_path = materialize_output_image(character.id, job.kind, job.runpod_job_id, state.output, extract_image_path(state.output))
        if preview_path:
            output_json["local_preview_path"] = preview_path
        job.output_json = dumps_json(output_json)
        job.error_message = state.error
        if state.status == "COMPLETED" and job.kind == "dataset":
            image_path = extract_image_path(state.output)
            local_path = preview_path
            if (image_path or local_path) and not _dataset_image_exists(db, job.id, image_path or local_path):
                remote_path = image_path
                caption_tags = normalize_tags((output_json.get("caption_tags") or "").split(","))
                db.add(
                    DatasetImage(
                        character_id=character.id,
                        generation_job_id=job.id,
                        image_path=local_path or cache_remote_reference(character.id, image_path),
                        remote_image_path=remote_path,
                        caption_tags_json=dumps_json(caption_tags),
                    )
                )
        db.add(job)
    db.commit()


def update_dataset_image(db: Session, dataset_image: DatasetImage, payload: DatasetImageUpdate) -> DatasetImage:
    dataset_image.keep_status = payload.keep_status
    dataset_image.notes = payload.notes.strip()
    dataset_image.caption_tags_json = dumps_json(normalize_tags(payload.caption_tags))
    db.add(dataset_image)
    db.commit()
    db.refresh(dataset_image)
    return dataset_image


def submit_training_run(db: Session, *, character: CharacterProfile, runpod_client, settings: Settings) -> TrainingRun:
    kept_images = [image for image in character.dataset_images if image.keep_status == "keep"]
    if len(kept_images) < 25:
        raise ValueError("Keep at least 25 images before training.")

    version = dataset_version()
    selected_images = [
        {
            "remote_image_path": image.remote_image_path,
            "caption_tags": loads_json(image.caption_tags_json, []),
        }
        for image in kept_images
        if image.remote_image_path
    ]
    if len(selected_images) < 25:
        raise ValueError("Kept images must have remote paths before training can start.")
    payload = {
        "character_id": character.id,
        "dataset_path": f"{settings.runpod_volume_root}/characters/{character.id}/dataset",
        "output_path": f"{settings.runpod_volume_root}/characters/{character.id}/loras/{version}",
        "base_model_ref": character.base_model_ref,
        "preset_name": TRAINING_PRESET["name"],
        "trigger_token": character.trigger_token,
        "selected_images": selected_images,
        "config": TRAINING_PRESET,
    }
    submission = runpod_client.submit_job(settings.runpod_train_endpoint_id, payload)
    training_run = TrainingRun(
        character_id=character.id,
        dataset_version=version,
        preset_name=TRAINING_PRESET["name"],
        kept_image_count=len(kept_images),
        runpod_job_id=submission.job_id,
        status=submission.status,
        config_json=dumps_json(TRAINING_PRESET),
        output_lora_path=str(submission.artifacts.get("lora_path") or ""),
        sample_previews_json=dumps_json(submission.artifacts.get("sample_previews") or []),
        logs_path=str(submission.artifacts.get("logs_path") or ""),
    )
    db.add(training_run)
    db.commit()
    db.refresh(training_run)
    return training_run


def sync_training_runs(db: Session, character: CharacterProfile, runpod_client, settings: Settings) -> None:
    pending = [run for run in character.training_runs if run.status not in {"COMPLETED", "FAILED", "CANCELLED"}]
    for run in pending:
        state = runpod_client.get_status(settings.runpod_train_endpoint_id, run.runpod_job_id)
        run.status = state.status
        run.error_message = state.error
        run.output_lora_path = str(state.output.get("lora_path") or run.output_lora_path)
        run.logs_path = str(state.output.get("logs_path") or run.logs_path)
        previews = state.output.get("sample_previews")
        if previews:
            run.sample_previews_json = dumps_json(previews)
        db.add(run)
    db.commit()


def submit_render(db: Session, *, character: CharacterProfile, request: RenderRequest, runpod_client, settings: Settings) -> GenerationJob:
    latest_completed_run = next((run for run in character.training_runs if run.status == "COMPLETED"), None)
    if latest_completed_run is None or not latest_completed_run.output_lora_path:
        raise ValueError("Train a LoRA successfully before rendering with consistency mode.")
    payload = {
        "mode": "render",
        "character_id": character.id,
        "base_model_ref": character.base_model_ref,
        "output_dir": f"{settings.runpod_volume_root}/characters/{character.id}/renders",
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "seed": request.seed,
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "cfg": request.cfg,
        "lora_path": latest_completed_run.output_lora_path,
        "lora_weight": request.lora_weight,
    }
    submission = runpod_client.submit_job(settings.runpod_generate_endpoint_id, payload)
    job = GenerationJob(
        character_id=character.id,
        kind="render",
        source_shot="render",
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
        width=request.width,
        height=request.height,
        steps=request.steps,
        cfg=request.cfg,
        lora_path=latest_completed_run.output_lora_path,
        lora_weight=request.lora_weight,
        runpod_job_id=submission.job_id,
        status=submission.status,
        output_json=dumps_json({"artifacts": submission.artifacts}),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def sync_character(db: Session, character: CharacterProfile, runpod_client, settings: Settings) -> CharacterProfile:
    sync_generation_jobs(db, character, runpod_client, settings)
    sync_training_runs(db, character, runpod_client, settings)
    refreshed = get_character(db, character.id)
    if refreshed is None:
        raise ValueError("Character not found after sync.")
    return refreshed


def kept_image_count(character: CharacterProfile) -> int:
    return sum(1 for image in character.dataset_images if image.keep_status == "keep")


def training_ready(character: CharacterProfile) -> bool:
    return kept_image_count(character) >= 25


def latest_completed_lora(character: CharacterProfile) -> str:
    for run in character.training_runs:
        if run.status == "COMPLETED" and run.output_lora_path:
            return run.output_lora_path
    return ""


def extract_image_path(output: dict) -> str:
    if isinstance(output.get("image_path"), str):
        return output["image_path"]
    images = output.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict):
            return str(first.get("path") or first.get("image_path") or "")
        if isinstance(first, str):
            return first
    return ""


def cache_remote_reference(character_id: int, remote_path: str) -> str:
    path = get_settings().data_dir / "characters" / str(character_id) / "remote_refs"
    ensure_dir(path)
    filename = remote_path.strip("/").replace("/", "__") or "artifact.txt"
    target = path / filename
    if not target.exists():
        target.write_text(remote_path, encoding="utf-8")
    return str(target)


def materialize_output_image(character_id: int, kind: str, job_id: str, output: dict, remote_path: str) -> str:
    if isinstance(output.get("image_base64"), str):
        return _write_b64_image(character_id, kind, job_id, output["image_base64"])
    images = output.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict) and isinstance(first.get("base64"), str):
            return _write_b64_image(character_id, kind, job_id, first["base64"])
    if remote_path.startswith("data/"):
        return remote_path
    return ""


def _write_b64_image(character_id: int, kind: str, job_id: str, payload: str) -> str:
    raw = payload.split(",", 1)[-1]
    image_dir = ensure_dir(get_settings().data_dir / "characters" / str(character_id) / f"{kind}_outputs")
    target = image_dir / f"{job_id}.png"
    if not target.exists():
        target.write_bytes(base64.b64decode(raw))
    return str(target)


def _dataset_image_exists(db: Session, generation_job_id: int, image_path: str) -> bool:
    stmt = select(DatasetImage).where(
        DatasetImage.generation_job_id == generation_job_id,
        DatasetImage.remote_image_path == image_path,
    )
    return db.scalar(stmt) is not None
