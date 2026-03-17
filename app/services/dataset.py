from __future__ import annotations

from datetime import datetime, timezone

from app.schemas import PromptPack
from app.services.utils import normalize_tags


TRAINING_PRESET = {
    "name": "anime_character_balanced",
    "resolution": "1024x1024",
    "rank": 16,
    "alpha": 16,
    "batch_size": 1,
    "grad_accum": 4,
    "optimizer": "AdamW8bit",
    "precision": "bf16",
    "max_train_steps": 1200,
    "sample_every": 200,
    "default_lora_weight": 0.8,
}


SHOT_MATRIX: list[dict[str, str]] = [
    {"label": "portrait-soft-smile-school", "framing": "portrait", "expression": "soft smile", "outfit": "school uniform", "background": "simple classroom blur"},
    {"label": "portrait-serious-school", "framing": "portrait", "expression": "serious", "outfit": "school uniform", "background": "simple classroom blur"},
    {"label": "portrait-joy-school", "framing": "portrait", "expression": "joyful grin", "outfit": "school uniform", "background": "simple classroom blur"},
    {"label": "portrait-shy-school", "framing": "portrait", "expression": "shy blush", "outfit": "school uniform", "background": "simple classroom blur"},
    {"label": "portrait-soft-smile-casual", "framing": "portrait", "expression": "soft smile", "outfit": "streetwear jacket", "background": "simple city lights blur"},
    {"label": "portrait-serious-casual", "framing": "portrait", "expression": "serious", "outfit": "streetwear jacket", "background": "simple city lights blur"},
    {"label": "portrait-joy-casual", "framing": "portrait", "expression": "joyful grin", "outfit": "streetwear jacket", "background": "simple city lights blur"},
    {"label": "portrait-thoughtful-casual", "framing": "portrait", "expression": "thoughtful", "outfit": "streetwear jacket", "background": "simple city lights blur"},
    {"label": "half-soft-smile-school", "framing": "half body", "expression": "soft smile", "outfit": "school uniform", "background": "simple campus path"},
    {"label": "half-serious-school", "framing": "half body", "expression": "serious", "outfit": "school uniform", "background": "simple campus path"},
    {"label": "half-joy-school", "framing": "half body", "expression": "joyful grin", "outfit": "school uniform", "background": "simple campus path"},
    {"label": "half-shy-school", "framing": "half body", "expression": "shy blush", "outfit": "school uniform", "background": "simple campus path"},
    {"label": "half-soft-smile-casual", "framing": "half body", "expression": "soft smile", "outfit": "streetwear jacket", "background": "simple city sidewalk"},
    {"label": "half-serious-casual", "framing": "half body", "expression": "serious", "outfit": "streetwear jacket", "background": "simple city sidewalk"},
    {"label": "half-joy-casual", "framing": "half body", "expression": "joyful grin", "outfit": "streetwear jacket", "background": "simple city sidewalk"},
    {"label": "half-thoughtful-casual", "framing": "half body", "expression": "thoughtful", "outfit": "streetwear jacket", "background": "simple city sidewalk"},
    {"label": "full-soft-smile-school", "framing": "full body", "expression": "soft smile", "outfit": "school uniform", "background": "simple rooftop sky"},
    {"label": "full-serious-school", "framing": "full body", "expression": "serious", "outfit": "school uniform", "background": "simple rooftop sky"},
    {"label": "full-joy-school", "framing": "full body", "expression": "joyful grin", "outfit": "school uniform", "background": "simple rooftop sky"},
    {"label": "full-shy-school", "framing": "full body", "expression": "shy blush", "outfit": "school uniform", "background": "simple rooftop sky"},
    {"label": "full-soft-smile-casual", "framing": "full body", "expression": "soft smile", "outfit": "streetwear jacket", "background": "simple downtown dusk"},
    {"label": "full-serious-casual", "framing": "full body", "expression": "serious", "outfit": "streetwear jacket", "background": "simple downtown dusk"},
    {"label": "full-joy-casual", "framing": "full body", "expression": "joyful grin", "outfit": "streetwear jacket", "background": "simple downtown dusk"},
    {"label": "full-thoughtful-casual", "framing": "full body", "expression": "thoughtful", "outfit": "streetwear jacket", "background": "simple downtown dusk"},
    {"label": "portrait-wink-school", "framing": "portrait", "expression": "playful wink", "outfit": "school uniform", "background": "simple window light"},
    {"label": "portrait-determined-casual", "framing": "portrait", "expression": "determined", "outfit": "streetwear jacket", "background": "simple evening neon"},
    {"label": "half-wink-school", "framing": "half body", "expression": "playful wink", "outfit": "school uniform", "background": "simple window light"},
    {"label": "half-determined-casual", "framing": "half body", "expression": "determined", "outfit": "streetwear jacket", "background": "simple evening neon"},
    {"label": "full-wink-school", "framing": "full body", "expression": "playful wink", "outfit": "school uniform", "background": "simple station platform"},
    {"label": "full-determined-casual", "framing": "full body", "expression": "determined", "outfit": "streetwear jacket", "background": "simple station platform"},
    {"label": "half-laugh-school", "framing": "half body", "expression": "laughing", "outfit": "school uniform", "background": "simple library aisle"},
    {"label": "half-laugh-casual", "framing": "half body", "expression": "laughing", "outfit": "streetwear jacket", "background": "simple cafe interior"},
]


def fallback_prompt_pack(name: str, trigger_token: str, visual_traits: str, default_outfit: str) -> PromptPack:
    return PromptPack(
        canonical_prompt=(
            f"{trigger_token}, 1girl, anime illustration, {name}, {visual_traits}, "
            f"wearing {default_outfit}, highly detailed, clean line art, soft cinematic shading"
        ),
        negative_prompt=(
            "lowres, blurry, bad anatomy, extra fingers, extra limbs, deformed hands, text, watermark, logo"
        ),
        style_tags=[
            "anime illustration",
            "nova anime xl",
            "illustrious style",
            "clean line art",
            "soft cinematic shading",
        ],
        shot_list=[item["label"] for item in SHOT_MATRIX],
    )


def build_dataset_prompts(prompt_pack: PromptPack, trigger_token: str, outfit_fallback: str) -> list[dict[str, str]]:
    style_suffix = ", ".join(prompt_pack.style_tags[:4])
    prompts: list[dict[str, str]] = []
    for item in SHOT_MATRIX:
        shot_prompt = (
            f"{prompt_pack.canonical_prompt}, {trigger_token}, {item['framing']}, {item['expression']}, "
            f"wearing {item['outfit'] or outfit_fallback}, {item['background']}, {style_suffix}"
        )
        prompts.append(
            {
                "label": item["label"],
                "prompt": shot_prompt,
                "caption_tags": ", ".join(
                    normalize_tags(
                        [
                            trigger_token,
                            item["framing"],
                            item["expression"],
                            item["outfit"] or outfit_fallback,
                            item["background"],
                            *prompt_pack.style_tags,
                        ]
                    )
                ),
            }
        )
    return prompts


def dataset_version() -> str:
    return datetime.now(timezone.utc).strftime("dataset-%Y%m%d-%H%M%S")
