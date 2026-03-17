# MVP Plan: Local Anime Character LoRA App

## Summary
- Build a local-only, single-user Python web app with a simple dark UI for one workflow: define an anime character, generate a synthetic dataset with Nova Anime XL, train a LoRA on RunPod Serverless, then render consistent images with that LoRA.
- Lock the image stack to **Nova Anime XL** and treat it as an **Illustrious-compatible SDXL-family workflow**, not a generic SD 1.5 setup.
- Use **Gemini `gemini-3-flash-preview`** only for prompt-pack generation and prompt refinement, not for image generation or training captions.

## Implementation Changes
- Local app stack: `FastAPI + Jinja2 + HTMX + SQLite`, dark theme, no auth, bind to `127.0.0.1` only, all metadata stored locally, images/artifacts stored under a local `data/` directory.
- Core records:
  - `CharacterProfile`: name, trigger token, visual traits, default outfit, canonical prompt, negative prompt, base model ref.
  - `GenerationJob`: prompt, seed, params, RunPod job id/status, output artifact paths.
  - `DatasetImage`: source generation id, caption tags, keep/discard flag, notes, split.
  - `TrainingRun`: dataset version, trainer preset, RunPod job id/status, output LoRA path, sample previews.
- Gemini service:
  - Input: structured character brief.
  - Output: strict JSON prompt pack with `canonical_prompt`, `negative_prompt`, `style_tags`, and a fixed `shot_list`.
  - Make model id configurable, default to `gemini-3-flash-preview`.
- RunPod generation service:
  - Use a **Flash queue endpoint** named `nova-generate`.
  - Load Nova Anime XL from a shared RunPod network volume.
  - Accept `prompt`, `negative_prompt`, `seed`, `width`, `height`, `steps`, `cfg`, optional `lora_path`, optional `lora_weight`.
  - Return image path plus full generation metadata.
- RunPod training service:
  - Use a **standard Serverless queue worker** named `lora-train` with a pinned Docker image and `kohya-ss/sd-scripts` SDXL LoRA training preset.
  - Read curated dataset from `/runpod-volume`, write LoRA weights, previews, and logs back to the same volume.
  - Keep training serverless, but do not force Flash for training; the pinned training environment is the safer MVP choice.
- Dataset workflow:
  - Synthetic-only.
  - Generate an initial **32-image shot matrix** across portrait / half-body / full-body, multiple expressions, 2 outfits, and simple backgrounds.
  - Derive training captions from normalized generation tags, always prefixing the character trigger token.
  - Let the user keep/discard images and edit tags manually.
  - Enable training only after **25-40 kept 1024px images** exist.
- Training/inference defaults:
  - One preset only: `anime_character_balanced`.
  - Fixed training params: `1024x1024`, `rank=16`, `alpha=16`, `batch_size=1`, `grad_accum=4`, `AdamW8bit`, `bf16`, `max_train_steps=1200`, sample preview every `200` steps.
  - Post-training inference uses the same `nova-generate` endpoint with the selected LoRA mounted at default weight `0.8`.

## Public Interfaces
- Local app routes:
  - `POST /api/characters`
  - `POST /api/characters/{id}/prompt-pack`
  - `POST /api/characters/{id}/dataset/generate`
  - `PATCH /api/dataset-images/{id}`
  - `POST /api/characters/{id}/train`
  - `POST /api/characters/{id}/render`
- RunPod payload contracts:
  - Generation request includes prompt params and optional LoRA fields.
  - Training request includes `character_id`, dataset path, base model ref, preset name, and output path.
  - Both services must return `job_id`, `status`, `submitted_at`, and artifact locations.

## Test Plan
- Unit tests for prompt-pack JSON validation, tag normalization, dataset state transitions, and RunPod/Gemini client adapters.
- Integration tests with mocked Gemini and RunPod responses for:
  - character creation -> prompt pack
  - batch dataset submission -> polling -> image registration
  - curation -> training submission -> status updates
  - LoRA-backed render submission
- Manual acceptance test:
  - Create one character, keep at least 25 generated images, finish one training run, then render 3 new images that preserve the same face/hair/clothing identity.

## Assumptions And Defaults
- The pasted Gemini key is treated as exposed; rotate it and use a replacement via `.env`, never hardcoded or stored in SQLite.
- Nova Anime XL is used for personal local work only; no public sharing, team auth, billing UI, or multi-user support in MVP.
- Generation/inference uses RunPod Flash because it reduces deployment friction; training stays on RunPod Serverless with Docker because the LoRA toolchain is dependency-sensitive.
- No automatic captioning model is added in MVP; captions come from normalized generation metadata plus manual edits, which is simpler and more consistent for a synthetic-only dataset.
