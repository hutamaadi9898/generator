# `nova-generate` endpoint

Implementation now lives in [runpod_workers/nova_generate/endpoint.py](/home/hutamaadi/Desktop/coding/generator/runpod_workers/nova_generate/endpoint.py).

What it does:

- RunPod Flash queue endpoint named `nova-generate`
- loads a Diffusers text-to-image pipeline on GPU
- resolves `base_model_ref` from `NOVA_MODEL_SOURCE` or `NOVA_MODEL_SOURCE_MAP_JSON`
- optionally loads a LoRA from `lora_path`
- writes a PNG plus sidecar JSON under the requested `output_dir`
- returns both `image_path` and `image_base64`

Required endpoint env:

- `RUNPOD_VOLUME_ROOT`
- `RUNPOD_VOLUME_NAME`
- `NOVA_MODEL_SOURCE` or `NOVA_MODEL_SOURCE_MAP_JSON`
- `CIVITAI_API_TOKEN` when `NOVA_MODEL_SOURCE` points at a Civitai page or download URL that needs auth
- `HF_TOKEN` when the base model is gated on Hugging Face

Model source examples:

- `NOVA_MODEL_SOURCE=https://civitai.com/models/376130/nova-anime-xl`
- `NOVA_MODEL_SOURCE=https://civitai.com/api/download/models/2741698`
- `NOVA_MODEL_SOURCE=/runpod-volume/models/novaAnimeXL_ilV170.safetensors`

Deploy from the repo root:

```bash
source .venv/bin/activate
pip install -e ".[runpod]"
flash env create prod
flash deploy --app nova-generate --env prod
```

This worker build is pinned to `runpod-flash==1.7.0`. If the endpoint was already
deployed with a different Flash package version, rebuild and redeploy it.
