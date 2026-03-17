from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.db import Base, engine, get_db
from app.models import DatasetImage
from app.schemas import CharacterBriefDraft, CharacterCreate, DatasetGenerateRequest, DatasetImageUpdate, RenderRequest
from app.services.gemini import GeminiClient
from app.services.orchestration import (
    apply_prompt_pack,
    create_character,
    get_character,
    kept_image_count,
    latest_completed_lora,
    list_characters,
    submit_dataset_generation,
    submit_render,
    submit_training_run,
    sync_character,
    training_ready,
    update_dataset_image,
)
from app.services.runpod import RunpodClient
from app.services.utils import loads_json
from app.settings import get_settings


settings = get_settings()
templates = Jinja2Templates(directory="app/templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    app.state.settings = settings
    app.state.gemini_client = GeminiClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        api_base=settings.gemini_api_base,
    )
    app.state.runpod_client = RunpodClient(
        api_key=settings.runpod_api_key,
        api_base=settings.runpod_api_base,
        submission_mode=settings.runpod_submission_mode,
        endpoint_function_names={
            settings.runpod_generate_endpoint_id: settings.runpod_generate_function_name,
            settings.runpod_train_endpoint_id: settings.runpod_train_function_name,
        },
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    app.mount("/data", StaticFiles(directory=str(settings.data_dir)), name="data")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
        characters = _load_dashboard(db, request.app)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=_template_context(
                request,
                characters=characters,
                brief_draft=_empty_brief_draft(),
                brief_flash_message="",
            ),
        )

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
        characters = _load_dashboard(db, request.app)
        return templates.TemplateResponse(
            request=request,
            name="partials/dashboard.html",
            context=_template_context(request, characters=characters),
        )

    @app.post("/api/characters", response_class=HTMLResponse)
    def create_character_route(
        request: Request,
        name: str = Form(...),
        trigger_token: str = Form(...),
        visual_traits: str = Form(...),
        default_outfit: str = Form(...),
        db: Session = Depends(get_db),
    ) -> HTMLResponse:
        try:
            payload = CharacterCreate(
                name=name,
                trigger_token=trigger_token,
                visual_traits=visual_traits,
                default_outfit=default_outfit,
            )
            create_character(
                db,
                name=payload.name,
                trigger_token=payload.trigger_token,
                visual_traits=payload.visual_traits,
                default_outfit=payload.default_outfit,
                base_model_ref=settings.base_model_ref,
            )
            return _dashboard_partial(request, db, "Character saved. Generate a prompt pack next.")
        except ValidationError as exc:
            return _dashboard_partial(request, db, _first_validation_error(exc))

    @app.post("/api/characters/autofill", response_class=HTMLResponse)
    def autofill_character_route(
        request: Request,
        name: str = Form(""),
        trigger_token: str = Form(""),
        visual_traits: str = Form(""),
        default_outfit: str = Form(""),
    ) -> HTMLResponse:
        gemini_client: GeminiClient = request.app.state.gemini_client
        draft = CharacterBriefDraft(
            name=name,
            trigger_token=trigger_token,
            visual_traits=visual_traits,
            default_outfit=default_outfit,
        )
        if not any([draft.name, draft.trigger_token, draft.visual_traits, draft.default_outfit]):
            message = "Add at least one clue before asking Gemini to fill the blanks."
            return _character_form_partial(request, draft, message)
        try:
            suggestion = gemini_client.autofill_character_brief(draft=draft)
            return _character_form_partial(request, suggestion, "Gemini filled the missing brief fields.")
        except Exception as exc:
            return _character_form_partial(request, draft, f"Gemini autofill failed: {exc}")

    @app.post("/api/characters/{character_id}/prompt-pack", response_class=HTMLResponse)
    def generate_prompt_pack_route(character_id: int, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
        character = _require_character(db, character_id)
        gemini_client: GeminiClient = request.app.state.gemini_client
        try:
            prompt_pack = gemini_client.generate_prompt_pack(
                name=character.name,
                trigger_token=character.trigger_token,
                visual_traits=character.visual_traits,
                default_outfit=character.default_outfit,
                base_model_ref=character.base_model_ref,
            )
            apply_prompt_pack(db, character, prompt_pack)
            return _dashboard_partial(request, db, f"Prompt pack ready with {len(prompt_pack.shot_list)} shots.")
        except Exception as exc:  # pragma: no cover - UI fallback
            return _dashboard_partial(request, db, f"Prompt pack failed: {exc}")

    @app.post("/api/characters/{character_id}/dataset/generate", response_class=HTMLResponse)
    def generate_dataset_route(
        character_id: int,
        request: Request,
        width: int = Form(1024),
        height: int = Form(1024),
        steps: int = Form(30),
        cfg: float = Form(6.0),
        seed_base: int = Form(1000),
        db: Session = Depends(get_db),
    ) -> HTMLResponse:
        character = _require_character(db, character_id)
        runpod_client: RunpodClient = request.app.state.runpod_client
        try:
            payload = DatasetGenerateRequest(width=width, height=height, steps=steps, cfg=cfg, seed_base=seed_base)
            submit_dataset_generation(db, character=character, request=payload, runpod_client=runpod_client, settings=settings)
            return _dashboard_partial(request, db, "Submitted 32 dataset generations to RunPod.")
        except Exception as exc:
            return _dashboard_partial(request, db, f"Dataset generation failed: {exc}")

    @app.patch("/api/dataset-images/{dataset_image_id}", response_class=HTMLResponse)
    def update_dataset_image_route(
        dataset_image_id: int,
        request: Request,
        keep_status: str = Form(...),
        notes: str = Form(""),
        caption_tags: str = Form(""),
        db: Session = Depends(get_db),
    ) -> HTMLResponse:
        image = db.get(DatasetImage, dataset_image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Dataset image not found.")
        try:
            payload = DatasetImageUpdate(
                keep_status=keep_status,
                notes=notes,
                caption_tags=[item.strip() for item in caption_tags.split(",") if item.strip()],
            )
            update_dataset_image(db, image, payload)
            return _dashboard_partial(request, db, "Dataset image updated.")
        except ValidationError as exc:
            return _dashboard_partial(request, db, _first_validation_error(exc))

    @app.post("/api/characters/{character_id}/train", response_class=HTMLResponse)
    def train_character_route(character_id: int, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
        character = _require_character(db, character_id)
        runpod_client: RunpodClient = request.app.state.runpod_client
        try:
            submit_training_run(db, character=character, runpod_client=runpod_client, settings=settings)
            return _dashboard_partial(request, db, "Training run submitted.")
        except Exception as exc:
            return _dashboard_partial(request, db, f"Training failed: {exc}")

    @app.post("/api/characters/{character_id}/render", response_class=HTMLResponse)
    def render_character_route(
        character_id: int,
        request: Request,
        prompt: str = Form(...),
        negative_prompt: str = Form(""),
        width: int = Form(1024),
        height: int = Form(1024),
        steps: int = Form(30),
        cfg: float = Form(6.0),
        lora_weight: float = Form(0.8),
        seed: int = Form(5000),
        db: Session = Depends(get_db),
    ) -> HTMLResponse:
        character = _require_character(db, character_id)
        runpod_client: RunpodClient = request.app.state.runpod_client
        try:
            payload = RenderRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                lora_weight=lora_weight,
                seed=seed,
            )
            submit_render(db, character=character, request=payload, runpod_client=runpod_client, settings=settings)
            return _dashboard_partial(request, db, "Render job submitted.")
        except ValidationError as exc:
            return _dashboard_partial(request, db, _first_validation_error(exc))
        except Exception as exc:
            return _dashboard_partial(request, db, f"Render failed: {exc}")

    return app


app = create_app()


def _dashboard_partial(request: Request, db: Session, flash_message: str) -> HTMLResponse:
    characters = _load_dashboard(db, request.app)
    return templates.TemplateResponse(
        request=request,
        name="partials/dashboard.html",
        context=_template_context(request, characters=characters, flash_message=flash_message),
    )


def _character_form_partial(request: Request, brief_draft: CharacterBriefDraft, flash_message: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="partials/character_form.html",
        context=_template_context(
            request,
            brief_draft=brief_draft,
            brief_flash_message=flash_message,
        ),
    )


def _load_dashboard(db: Session, app: FastAPI) -> list[dict[str, Any]]:
    db.expire_all()
    rows = list_characters(db)
    runpod_client: RunpodClient = app.state.runpod_client
    synced: list[dict[str, Any]] = []
    for row in rows:
        try:
            if row.generation_jobs or row.training_runs:
                row = sync_character(db, row, runpod_client, settings)
        except Exception:
            pass
        synced.append(_serialize_character(row))
    return synced


def _serialize_character(character) -> dict[str, Any]:
    dataset_images = [
        {
            "id": image.id,
            "image_path": image.image_path,
            "public_path": to_public_path(image.image_path),
            "remote_image_path": image.remote_image_path,
            "caption_tags": loads_json(image.caption_tags_json, []),
            "keep_status": image.keep_status,
            "notes": image.notes,
            "created_at": image.created_at,
        }
        for image in character.dataset_images[:12]
    ]
    generation_jobs = [
        {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "prompt": job.prompt,
            "source_shot": job.source_shot,
            "seed": job.seed,
            "preview_path": to_public_path(loads_json(job.output_json, {}).get("local_preview_path", "")),
            "error_message": job.error_message,
            "created_at": job.created_at,
        }
        for job in character.generation_jobs[:10]
    ]
    training_runs = [
        {
            "id": run.id,
            "status": run.status,
            "dataset_version": run.dataset_version,
            "kept_image_count": run.kept_image_count,
            "output_lora_path": run.output_lora_path,
            "sample_previews": loads_json(run.sample_previews_json, []),
            "logs_path": run.logs_path,
            "error_message": run.error_message,
            "created_at": run.created_at,
        }
        for run in character.training_runs[:6]
    ]
    return {
        "id": character.id,
        "name": character.name,
        "trigger_token": character.trigger_token,
        "visual_traits": character.visual_traits,
        "default_outfit": character.default_outfit,
        "canonical_prompt": character.canonical_prompt,
        "negative_prompt": character.negative_prompt,
        "style_tags": loads_json(character.style_tags_json, []),
        "shot_list": loads_json(character.shot_list_json, []),
        "base_model_ref": character.base_model_ref,
        "dataset_images": dataset_images,
        "generation_jobs": generation_jobs,
        "training_runs": training_runs,
        "kept_image_count": kept_image_count(character),
        "training_ready": training_ready(character),
        "latest_lora": latest_completed_lora(character),
        "created_at": character.created_at,
    }


def _template_context(request: Request, **kwargs: Any) -> dict[str, Any]:
    return {
        "request": request,
        "app_name": settings.app_name,
        "gemini_model": settings.gemini_model,
        "base_model_ref": settings.base_model_ref,
        "current_year": datetime.now(timezone.utc).year,
        **kwargs,
    }


def _require_character(db: Session, character_id: int):
    character = get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found.")
    return character


def _first_validation_error(exc: ValidationError) -> str:
    return exc.errors()[0]["msg"]


def to_public_path(path: str) -> str:
    if not path:
        return ""
    data_prefix = f"{settings.data_dir.as_posix()}/"
    if path.startswith(data_prefix):
        return "/data/" + path.removeprefix(data_prefix)
    return path


def _empty_brief_draft() -> CharacterBriefDraft:
    return CharacterBriefDraft(name="", trigger_token="", visual_traits="", default_outfit="")
