from __future__ import annotations

import json
from typing import Any

import httpx

from app.schemas import CharacterBriefDraft, CharacterBriefSuggestion, PromptPack
from app.services.dataset import fallback_prompt_pack


PROMPT_PACK_SCHEMA = {
    "type": "object",
    "properties": {
        "canonical_prompt": {"type": "string"},
        "negative_prompt": {"type": "string"},
        "style_tags": {"type": "array", "items": {"type": "string"}},
        "shot_list": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["canonical_prompt", "negative_prompt", "style_tags", "shot_list"],
}

BRIEF_AUTOFILL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "trigger_token": {"type": "string"},
        "visual_traits": {"type": "string"},
        "default_outfit": {"type": "string"},
    },
    "required": ["name", "trigger_token", "visual_traits", "default_outfit"],
}


class GeminiClient:
    def __init__(self, api_key: str, model: str, api_base: str, client: httpx.Client | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.client = client or httpx.Client(timeout=60)

    def generate_prompt_pack(
        self,
        *,
        name: str,
        trigger_token: str,
        visual_traits: str,
        default_outfit: str,
        base_model_ref: str,
    ) -> PromptPack:
        if not self.api_key:
            return fallback_prompt_pack(name, trigger_token, visual_traits, default_outfit)

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Return JSON only. Build a concise prompt pack for an anime LoRA dataset. "
                                "The base image model is Nova Anime XL on the Illustrious SDXL family. "
                                f"Character name: {name}. Trigger token: {trigger_token}. "
                                f"Visual traits: {visual_traits}. Default outfit: {default_outfit}. "
                                f"Base model ref: {base_model_ref}. "
                                "The shot_list must contain exactly 32 short shot labels suitable for dataset generation."
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": PROMPT_PACK_SCHEMA,
                "temperature": 0.4,
            },
        }
        response = self.client.post(
            f"{self.api_base}/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        text = self._extract_text(body)
        data = json.loads(text)
        return PromptPack.model_validate(data)

    def autofill_character_brief(self, *, draft: CharacterBriefDraft) -> CharacterBriefSuggestion:
        if not self.api_key:
            return self._fallback_brief(draft)

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Return JSON only. Complete missing fields for an anime original character brief. "
                                "Preserve any non-empty field exactly as provided. Only invent values for blank fields. "
                                "Keep visual_traits as a compact comma-separated list. "
                                "Keep default_outfit as one concise outfit phrase. "
                                f"Current draft: name={draft.name!r}, trigger_token={draft.trigger_token!r}, "
                                f"visual_traits={draft.visual_traits!r}, default_outfit={draft.default_outfit!r}."
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": BRIEF_AUTOFILL_SCHEMA,
                "temperature": 0.5,
            },
        }
        response = self.client.post(
            f"{self.api_base}/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        text = self._extract_text(body)
        data = json.loads(text)
        suggestion = CharacterBriefSuggestion.model_validate(data)
        return CharacterBriefSuggestion(
            name=draft.name or suggestion.name,
            trigger_token=draft.trigger_token or suggestion.trigger_token,
            visual_traits=draft.visual_traits or suggestion.visual_traits,
            default_outfit=draft.default_outfit or suggestion.default_outfit,
        )

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        candidates = body.get("candidates") or []
        if not candidates:
            raise ValueError("Gemini response did not include candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if "text" in part:
                return part["text"]
        raise ValueError("Gemini response did not include text content.")

    @staticmethod
    def _fallback_brief(draft: CharacterBriefDraft) -> CharacterBriefSuggestion:
        base_name = draft.name or "Kimi Hime"
        trigger_token = draft.trigger_token or base_name.strip().lower().replace(" ", "_")
        visual_traits = draft.visual_traits or "long silver hair, teal eyes, calm expression, slim silhouette, sharp bangs"
        default_outfit = draft.default_outfit or "tailored black school blazer with teal ribbon"
        return CharacterBriefSuggestion(
            name=base_name,
            trigger_token=trigger_token,
            visual_traits=visual_traits,
            default_outfit=default_outfit,
        )
