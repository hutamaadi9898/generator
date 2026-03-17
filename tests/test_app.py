from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import json
from pathlib import Path

import cloudpickle
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import app.main as app_module
from app.db import Base, get_db
from app.main import app
from app.models import CharacterProfile, DatasetImage, TrainingRun
from app.schemas import CharacterBriefDraft, CharacterBriefSuggestion, PromptPack
from app.services.dataset import SHOT_MATRIX, fallback_prompt_pack
from app.services.gemini import GeminiClient
from app.services.orchestration import training_ready
from app.services.runpod import RunpodClient
from app.settings import get_settings


PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WnM9L8AAAAASUVORK5CYII="


class FakeGeminiClient:
    def autofill_character_brief(self, *, draft: CharacterBriefDraft) -> CharacterBriefSuggestion:
        return CharacterBriefSuggestion(
            name=draft.name or "Kimi Hime",
            trigger_token=draft.trigger_token or "kimi_hime",
            visual_traits=draft.visual_traits or "long silver hair, teal eyes, calm expression, slim silhouette, sharp bangs",
            default_outfit=draft.default_outfit or "tailored black school blazer with teal ribbon",
        )

    def generate_prompt_pack(self, *, name: str, trigger_token: str, visual_traits: str, default_outfit: str, base_model_ref: str) -> PromptPack:
        pack = fallback_prompt_pack(name, trigger_token, visual_traits, default_outfit)
        return PromptPack(
            canonical_prompt=pack.canonical_prompt,
            negative_prompt=pack.negative_prompt,
            style_tags=pack.style_tags,
            shot_list=[item["label"] for item in SHOT_MATRIX],
        )


class FakeRunpodClient:
    def __init__(self) -> None:
        self.counter = 0
        self.jobs: dict[str, dict] = {}

    def submit_job(self, endpoint_id: str, payload: dict) -> object:
        self.counter += 1
        job_id = f"job-{self.counter}"
        if payload.get("preset_name") == "anime_character_balanced":
            self.jobs[job_id] = {
                "status": "COMPLETED",
                "output": {
                    "lora_path": f"/runpod-volume/characters/{payload['character_id']}/loras/final.safetensors",
                    "logs_path": f"/runpod-volume/characters/{payload['character_id']}/logs/train.log",
                    "sample_previews": [f"/runpod-volume/characters/{payload['character_id']}/previews/preview-1.png"],
                },
            }
        else:
            self.jobs[job_id] = {
                "status": "COMPLETED",
                "output": {
                    "image_path": f"/runpod-volume/characters/{payload['character_id']}/{payload['mode']}/{job_id}.png",
                    "image_base64": PNG_B64,
                },
            }
        return type(
            "Submission",
            (),
            {
                "job_id": job_id,
                "status": "IN_QUEUE",
                "submitted_at": "2026-03-17T00:00:00Z",
                "artifacts": {},
            },
        )()

    def get_status(self, endpoint_id: str, job_id: str) -> object:
        state = self.jobs[job_id]
        return type(
            "State",
            (),
            {
                "job_id": job_id,
                "status": state["status"],
                "output": state["output"],
                "error": "",
            },
        )()


@pytest.fixture()
def client(tmp_path: Path):
    settings = get_settings()
    original_data_dir = settings.data_dir
    original_engine = app_module.engine
    original_lifespan = app.router.lifespan_context
    settings.data_dir = tmp_path / "data"
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    db_file = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    Base.metadata.create_all(bind=engine)
    app_module.engine = engine

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    app.router.lifespan_context = noop_lifespan

    def override_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db

    with TestClient(app) as test_client:
        test_client.app.state.gemini_client = FakeGeminiClient()
        test_client.app.state.runpod_client = FakeRunpodClient()
        yield test_client, TestingSessionLocal

    app.dependency_overrides.clear()
    app_module.engine = original_engine
    app.router.lifespan_context = original_lifespan
    settings.data_dir = original_data_dir


def test_gemini_client_parses_json_payload() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{"canonical_prompt":"hero prompt","negative_prompt":"bad anatomy",'
                                '"style_tags":["anime illustration"],"shot_list":["portrait-1"]}'
                            )
                        }
                    ]
                }
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(":generateContent")
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = GeminiClient(
        api_key="secret",
        model="gemini-3-flash-preview",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        client=httpx.Client(transport=transport),
    )

    prompt_pack = client.generate_prompt_pack(
        name="Aoi",
        trigger_token="aoi_token",
        visual_traits="silver hair, teal eyes",
        default_outfit="school uniform",
        base_model_ref="nova-anime-xl-illustrious",
    )

    assert prompt_pack.canonical_prompt == "hero prompt"
    assert prompt_pack.style_tags == ["anime illustration"]


def test_gemini_client_autofills_missing_brief_fields() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{"name":"Kimi Hime","trigger_token":"kimi_hime",'
                                '"visual_traits":"silver hair, teal eyes","default_outfit":"black blazer with teal ribbon"}'
                            )
                        }
                    ]
                }
            }
        ]
    }

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    client = GeminiClient(
        api_key="secret",
        model="gemini-3-flash-preview",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        client=httpx.Client(transport=transport),
    )

    suggestion = client.autofill_character_brief(
        draft=CharacterBriefDraft(name="Kimi Hime", trigger_token="", visual_traits="", default_outfit="")
    )

    assert suggestion.name == "Kimi Hime"
    assert suggestion.trigger_token == "kimi_hime"
    assert suggestion.visual_traits == "silver hair, teal eyes"


def test_runpod_client_extracts_submission() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert json.loads(request.content) == {"input": {"prompt": "test"}}
            return httpx.Response(200, json={"id": "abc123", "status": "IN_QUEUE", "output": {"foo": "bar"}})
        return httpx.Response(200, json={"id": "abc123", "status": "COMPLETED", "output": {"image_path": "/tmp/out.png"}})

    transport = httpx.MockTransport(handler)
    client = RunpodClient(
        api_key="secret",
        api_base="https://api.runpod.ai/v2",
        client=httpx.Client(transport=transport),
        submission_mode="plain",
    )

    submission = client.submit_job("endpoint-1", {"prompt": "test"})
    status = client.get_status("endpoint-1", "abc123")

    assert submission.job_id == "abc123"
    assert status.status == "COMPLETED"
    assert status.output["image_path"] == "/tmp/out.png"


def test_runpod_client_handles_flash_function_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["input"]["function_name"] == "nova_generate"
            assert body["input"]["execution_type"] == "function"
            assert body["input"]["serialization_format"] == "cloudpickle"
            assert cloudpickle.loads(base64.b64decode(body["input"]["kwargs"]["prompt"])) == "test"
            return httpx.Response(
                200,
                json={
                    "id": "abc123",
                    "status": "IN_QUEUE",
                    "output": {
                        "success": True,
                        "result": base64.b64encode(cloudpickle.dumps({"foo": "bar"})).decode("utf-8"),
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "abc123",
                "status": "COMPLETED",
                "output": {
                    "success": True,
                    "result": base64.b64encode(cloudpickle.dumps({"image_path": "/tmp/out.png"})).decode("utf-8"),
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = RunpodClient(
        api_key="secret",
        api_base="https://api.runpod.ai/v2",
        client=httpx.Client(transport=transport),
        submission_mode="flash_function",
        endpoint_function_names={"endpoint-1": "nova_generate"},
    )

    submission = client.submit_job("endpoint-1", {"prompt": "test"})
    status = client.get_status("endpoint-1", "abc123")

    assert submission.artifacts["foo"] == "bar"
    assert status.output["image_path"] == "/tmp/out.png"


def test_full_character_flow(client) -> None:
    test_client, session_factory = client

    response = test_client.post(
        "/api/characters",
        data={
            "name": "Aoi Kisaragi",
            "trigger_token": "aoi_kisaragi",
            "visual_traits": "long silver hair, teal eyes, calm expression, sharp bangs",
            "default_outfit": "tailored black school blazer with teal ribbon",
        },
    )
    assert response.status_code == 200
    assert "Character saved" in response.text

    with session_factory() as db:
        character = db.scalar(select(CharacterProfile))
        assert character is not None
        character_id = character.id

    response = test_client.post(f"/api/characters/{character_id}/prompt-pack")
    assert response.status_code == 200
    assert "Prompt pack ready" in response.text

    response = test_client.post(
        f"/api/characters/{character_id}/dataset/generate",
        data={"width": 1024, "height": 1024, "steps": 30, "cfg": 6.0, "seed_base": 1000},
    )
    assert response.status_code == 200
    assert "Submitted 32 dataset generations" in response.text

    with session_factory() as db:
        images = list(db.scalars(select(DatasetImage).order_by(DatasetImage.id)))
        assert len(images) == 32
        for image in images[:25]:
            image.keep_status = "keep"
        db.commit()

        character = db.get(CharacterProfile, character_id)
        assert character is not None
        db.refresh(character)

    response = test_client.get("/dashboard")
    assert response.status_code == 200

    response = test_client.post(f"/api/characters/{character_id}/train")
    assert response.status_code == 200
    assert "Training run submitted" in response.text

    with session_factory() as db:
        run = db.scalar(select(TrainingRun))
        assert run is not None
        run.status = "COMPLETED"
        run.output_lora_path = "/runpod-volume/characters/1/loras/final.safetensors"
        db.commit()

    response = test_client.post(
        f"/api/characters/{character_id}/render",
        data={
            "prompt": "aoi_kisaragi, anime illustration, neon alley, intense eye contact",
            "negative_prompt": "bad anatomy",
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "cfg": 6.0,
            "lora_weight": 0.8,
            "seed": 5000,
        },
    )
    assert response.status_code == 200
    assert "Render job submitted" in response.text

    with session_factory() as db:
        character = db.get(CharacterProfile, character_id)
        assert character is not None
        assert training_ready(character)


def test_autofill_route_updates_form_fields(client) -> None:
    test_client, _ = client

    response = test_client.post(
        "/api/characters/autofill",
        data={
            "name": "Kimi Hime",
            "trigger_token": "",
            "visual_traits": "",
            "default_outfit": "",
        },
    )

    assert response.status_code == 200
    assert 'value="Kimi Hime"' in response.text
    assert 'value="kimi_hime"' in response.text
    assert "Gemini filled the missing brief fields." in response.text
