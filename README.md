# Anime LoRA Lab

Local FastAPI app for building a consistent anime character workflow:

- create a character brief
- generate a Gemini prompt pack
- submit a 32-shot synthetic dataset batch to RunPod
- curate keep/discard dataset images
- submit a LoRA training run
- render new images with the finished LoRA

## Stack

- FastAPI + Jinja2 + HTMX
- SQLite
- Gemini `gemini-3-flash-preview` for prompt packs
- RunPod Serverless GPU endpoints for generation and LoRA training

## Quick Start

1. Create a virtualenv and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

2. Copy the env file and rotate the Gemini key before using it.

```bash
cp .env.example .env
```

3. Run the app.

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

4. Open `http://127.0.0.1:8000`.

For the optional legacy Flash path, install the pinned Flash dependency:

```bash
pip install -e ".[dev,runpod]"
```

The recommended production path is now a custom RunPod Serverless image built
from [`Dockerfile.serverless`](/home/hutamaadi/Desktop/coding/generator/Dockerfile.serverless)
and deployed through RunPod's GitHub integration. That avoids the Flash build
artifact limit entirely.

## Environment

Required variables:

- `GEMINI_API_KEY`
- `RUNPOD_API_KEY`
- `RUNPOD_GENERATE_ENDPOINT_ID`
- `RUNPOD_TRAIN_ENDPOINT_ID`

Important defaults:

- `BASE_MODEL_REF=nova-anime-xl-illustrious`
- `GEMINI_MODEL=gemini-3-flash-preview`
- `RUNPOD_SUBMISSION_MODE=plain`
- `RUNPOD_GENERATE_FUNCTION_NAME=nova_generate`
- `RUNPOD_TRAIN_FUNCTION_NAME=lora_train`
- `RUNPOD_VOLUME_ROOT=/runpod-volume`

Worker deployment variables:

- `RUNPOD_VOLUME_NAME=anime-lora-lab`
- `RUNPOD_VOLUME_SIZE_GB=300`
- `CIVITAI_API_TOKEN=...` when downloading Nova directly from Civitai inside the worker
- `HF_TOKEN=...` when using a gated model repo
- `NOVA_MODEL_SOURCE=...` to map the app alias `nova-anime-xl-illustrious` to a real local path, URL, Civitai model page, Civitai download URL, or Hugging Face repo ID
- `NOVA_SINGLE_FILE_PIPELINE_CLASS=StableDiffusionXLPipeline` to force the concrete diffusers pipeline class used for `.safetensors` or `.ckpt` single-file checkpoints when auto-detection is not enough
- `SD_SCRIPTS_REPO=https://github.com/kohya-ss/sd-scripts.git`
- `SD_SCRIPTS_REF=main`

## RunPod Contracts

The local app expects two remote services.

### Generation endpoint

Request shape sent to the worker function:

```json
{
  "mode": "dataset",
  "character_id": 1,
  "base_model_ref": "nova-anime-xl-illustrious",
  "output_dir": "/runpod-volume/characters/1/dataset",
  "prompt": "prompt text",
  "negative_prompt": "negative text",
  "seed": 12345,
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "cfg": 6.0,
  "lora_path": "/runpod-volume/characters/1/loras/final.safetensors",
  "lora_weight": 0.8,
  "metadata": {
    "shot_label": "portrait-soft-smile-school",
    "caption_tags": "aoi_kisaragi, portrait, soft smile"
  }
}
```

When `RUNPOD_SUBMISSION_MODE=plain`, the app sends that JSON payload directly to RunPod.

Legacy Flash compatibility is still available through `RUNPOD_SUBMISSION_MODE=flash_function`,
which wraps the payload like this:

```json
{
  "function_name": "nova_generate",
  "execution_type": "function",
  "serialization_format": "cloudpickle",
  "kwargs": {
    "...": "base64-encoded cloudpickle values"
  }
}
```

Accepted completion payload:

```json
{
  "image_path": "/runpod-volume/characters/1/dataset/job-1.png",
  "image_base64": "optional_base64_png"
}
```

`image_base64` is optional but strongly recommended so the local dashboard can show previews immediately.

### Training endpoint

Request shape sent to the worker function:

```json
{
  "character_id": 1,
  "dataset_path": "/runpod-volume/characters/1/dataset",
  "output_path": "/runpod-volume/characters/1/loras/dataset-20260317-120000",
  "base_model_ref": "nova-anime-xl-illustrious",
  "preset_name": "anime_character_balanced",
  "trigger_token": "aoi_kisaragi",
  "selected_images": [
    {
      "remote_image_path": "/runpod-volume/characters/1/dataset/job-1.png",
      "caption_tags": ["aoi_kisaragi", "portrait", "soft smile"]
    }
  ],
  "config": {
    "resolution": "1024x1024",
    "rank": 16,
    "alpha": 16,
    "batch_size": 1,
    "grad_accum": 4,
    "optimizer": "AdamW8bit",
    "precision": "bf16",
    "max_train_steps": 1200,
    "sample_every": 200,
    "default_lora_weight": 0.8
  }
}
```

When `RUNPOD_SUBMISSION_MODE=plain`, the app sends that JSON payload directly to RunPod.

Legacy Flash compatibility is still available through `RUNPOD_SUBMISSION_MODE=flash_function`,
which wraps the payload like this:

```json
{
  "function_name": "lora_train",
  "execution_type": "function",
  "serialization_format": "cloudpickle",
  "kwargs": {
    "...": "base64-encoded cloudpickle values"
  }
}
```

Accepted completion payload:

```json
{
  "lora_path": "/runpod-volume/characters/1/loras/final.safetensors",
  "logs_path": "/runpod-volume/characters/1/logs/train.log",
  "sample_previews": [
    "/runpod-volume/characters/1/previews/sample-1.png"
  ]
}
```

## Serverless GitHub Deploy

The repo now includes a plain RunPod Serverless worker entrypoint:

- [runpod_serverless/main.py](/home/hutamaadi/Desktop/coding/generator/runpod_serverless/main.py)
- [Dockerfile.serverless](/home/hutamaadi/Desktop/coding/generator/Dockerfile.serverless)
- [requirements.serverless.txt](/home/hutamaadi/Desktop/coding/generator/requirements.serverless.txt)

Recommended setup:

1. In RunPod, create two Serverless endpoints from this GitHub repo and `Dockerfile.serverless`.
2. Set `RUNPOD_HANDLER=nova_generate` on the generation endpoint.
3. Set `RUNPOD_HANDLER=lora_train` on the training endpoint.
4. Mount the same network volume on both endpoints.
5. Set `RUNPOD_SUBMISSION_MODE=plain` in the app.

This repo includes:

- a CI workflow at [.github/workflows/ci.yml](/home/hutamaadi/Desktop/coding/generator/.github/workflows/ci.yml)
- a release workflow at [.github/workflows/serverless-release.yml](/home/hutamaadi/Desktop/coding/generator/.github/workflows/serverless-release.yml)
- deployment notes at [docs/runpod-serverless-github.md](/home/hutamaadi/Desktop/coding/generator/docs/runpod-serverless-github.md)

Push a tag like `serverless-v2026.03.17.1` and the release workflow will create
a GitHub release for RunPod to pick up.

## Legacy Flash Endpoints

The repo now includes two deployable Flash workers:

- [runpod_workers/nova_generate/endpoint.py](/home/hutamaadi/Desktop/coding/generator/runpod_workers/nova_generate/endpoint.py)
- [runpod_workers/lora_train/endpoint.py](/home/hutamaadi/Desktop/coding/generator/runpod_workers/lora_train/endpoint.py)

Install Flash tooling locally:

```bash
source .venv/bin/activate
pip install -e ".[dev,runpod]"
```

Typical legacy deployment flow:

```bash
source .venv/bin/activate
export RUNPOD_API_KEY=...
flash env create prod
flash deploy --app nova-generate --env prod
flash deploy --app lora-train --env prod
```

For local Docker preview on older NVIDIA drivers, use the wrapper instead of
calling `flash` directly:

```bash
source .venv/bin/activate
./scripts/flash_local.sh deploy --preview
```

After deploy, copy the generated endpoint IDs into:

- `RUNPOD_GENERATE_ENDPOINT_ID`
- `RUNPOD_TRAIN_ENDPOINT_ID`

Notes:

- `nova-anime-xl-illustrious` is treated as an app-level alias. The worker resolves it through `NOVA_MODEL_SOURCE` or `NOVA_MODEL_SOURCE_MAP_JSON`.
- As of March 17, 2026, the Civitai model page [Nova Anime XL](https://civitai.com/models/376130/nova-anime-xl) reports the latest Illustrious build as `IL V17.0`, published on March 3, 2026. The worker can resolve the model page URL directly, so you do not need to hardcode the `api/download/models/...` URL unless you want it pinned.
- The training worker assumes an SDXL-compatible checkpoint and launches `sdxl_train_network.py`.
- Both workers require the same shared RunPod network volume so dataset images, LoRAs, and cached models are visible to both endpoints.

### Local Preview and CUDA Driver Mismatch

If local preview fails with an error like `cuda>=12.8`, the problem is the
local Docker image, not the worker code. `runpod-flash` defaults to
`runpod/flash:latest`, and Docker Hub `latest` can move to a newer CUDA base
than your installed NVIDIA driver supports.

This repo includes [`scripts/flash_local.sh`](/home/hutamaadi/Desktop/coding/generator/scripts/flash_local.sh),
which pins `FLASH_IMAGE_TAG=1.0.1` and uses fully qualified `docker.io/runpod/...`
image names for the local command only:

```bash
./scripts/flash_local.sh deploy --preview
```

By default the wrapper binds the preview load balancer to port `18000` so it
does not collide with the local FastAPI app on `8000`.

If you want the plain `flash` command to use the same pin, set this in `.env`:

```bash
FLASH_IMAGE_TAG=1.0.1
FLASH_GPU_IMAGE=docker.io/runpod/flash:1.0.1
FLASH_CPU_IMAGE=docker.io/runpod/flash-cpu:1.0.1
FLASH_LB_IMAGE=docker.io/runpod/flash-lb:1.0.1
FLASH_CPU_LB_IMAGE=docker.io/runpod/flash-lb-cpu:1.0.1
FLASH_BUILD_PYTHON_VERSION=3.12
```

If you would rather stay on Docker Hub `latest`, update the local NVIDIA driver
to a CUDA 12.8-compatible release first.

## Tests

```bash
pytest
```

## Notes

- The app is single-user and local-only.
- Prompt packs fall back to a deterministic local template when `GEMINI_API_KEY` is missing.
- Dataset generation is blocked until a prompt pack exists.
- Training is blocked until at least 25 images are marked `keep`.
