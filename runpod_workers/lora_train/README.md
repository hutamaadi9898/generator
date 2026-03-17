# `lora-train` endpoint

Implementation now lives in [runpod_workers/lora_train/endpoint.py](/home/hutamaadi/Desktop/coding/generator/runpod_workers/lora_train/endpoint.py).

What it does:

- RunPod Flash queue endpoint named `lora-train`
- clones or updates `kohya-ss/sd-scripts` onto the shared volume
- materializes the curated `selected_images` list into a Kohya-style dataset folder
- resolves the base model locally before launching `sdxl_train_network.py`
- writes weights, logs, manifests, and sample previews back under `output_path`

Required endpoint env:

- `RUNPOD_VOLUME_ROOT`
- `RUNPOD_VOLUME_NAME`
- `NOVA_MODEL_SOURCE` or `NOVA_MODEL_SOURCE_MAP_JSON`
- `CIVITAI_API_TOKEN` when `NOVA_MODEL_SOURCE` points at a Civitai page or download URL that needs auth
- `SD_SCRIPTS_REPO`
- `SD_SCRIPTS_REF`
- `HF_TOKEN` when the base model is gated on Hugging Face

Deploy from the repo root:

```bash
source .venv/bin/activate
pip install -e ".[runpod]"
flash deploy --app lora-train --env prod
```

This worker build is pinned to `runpod-flash==1.7.0`. If the endpoint was already
deployed with a different Flash package version, rebuild and redeploy it.
