from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CharacterCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    trigger_token: str = Field(min_length=2, max_length=80)
    visual_traits: str = Field(min_length=10)
    default_outfit: str = Field(min_length=3)

    @field_validator("trigger_token")
    @classmethod
    def normalize_trigger_token(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")


class CharacterBriefDraft(BaseModel):
    name: str = ""
    trigger_token: str = ""
    visual_traits: str = ""
    default_outfit: str = ""

    @field_validator("name", "trigger_token", "visual_traits", "default_outfit", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str:
        return (value or "").strip()


class CharacterBriefSuggestion(CharacterBriefDraft):
    pass


class PromptPack(BaseModel):
    canonical_prompt: str
    negative_prompt: str
    style_tags: list[str]
    shot_list: list[str]


class GenerationDefaults(BaseModel):
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg: float = 6.0


class DatasetGenerateRequest(BaseModel):
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg: float = 6.0
    seed_base: int = 1000


class DatasetImageUpdate(BaseModel):
    keep_status: str = Field(pattern="^(pending|keep|discard)$")
    notes: str = ""
    caption_tags: list[str] = Field(default_factory=list)


class RenderRequest(BaseModel):
    prompt: str = Field(min_length=10)
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg: float = 6.0
    lora_weight: float = 0.8
    seed: int = 5000


@dataclass(slots=True)
class RunpodSubmission:
    job_id: str
    status: str
    submitted_at: str | None
    artifacts: dict[str, Any]


@dataclass(slots=True)
class RunpodJobState:
    job_id: str
    status: str
    output: dict[str, Any]
    error: str = ""
