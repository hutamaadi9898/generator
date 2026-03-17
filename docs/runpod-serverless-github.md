# RunPod Serverless GitHub Deploy

This repo now supports a plain RunPod Serverless deployment path that does not
depend on Flash build artifacts.

## Files

- `Dockerfile.serverless`
- `runpod_serverless/main.py`
- `requirements.serverless.txt`

## Recommended endpoint setup

Create two RunPod Serverless endpoints from the same GitHub repo and Dockerfile:

1. Generation endpoint
   Set `RUNPOD_HANDLER=nova_generate`
2. Training endpoint
   Set `RUNPOD_HANDLER=lora_train`

Both endpoints should mount the same network volume and pass the same shared
environment variables used by the app, especially:

- `RUNPOD_VOLUME_ROOT=/runpod-volume`
- `NOVA_MODEL_SOURCE=...`
- `HF_TOKEN=...` when needed
- `CIVITAI_API_TOKEN=...` when needed
- `SD_SCRIPTS_REPO=...`
- `SD_SCRIPTS_REF=...`

## GitHub release flow

RunPod's GitHub integration rebuilds from GitHub releases. This repo includes a
workflow that creates a GitHub release when you push a tag matching
`serverless-v*`.

Example:

```bash
git tag serverless-v2026.03.17.1
git push origin serverless-v2026.03.17.1
```

## App configuration

Use plain JSON submission mode in the local app:

```env
RUNPOD_SUBMISSION_MODE=plain
```
