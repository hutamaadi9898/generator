from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_VOLUME_NAME = os.getenv("RUNPOD_VOLUME_NAME", "anime-lora-lab")
DEFAULT_VOLUME_SIZE_GB = int(os.getenv("RUNPOD_VOLUME_SIZE_GB", "300"))
DEFAULT_DATA_CENTER = os.getenv("RUNPOD_DATA_CENTER", "EU_RO_1")
DEFAULT_MODEL_ALIAS = "nova-anime-xl-illustrious"
DEFAULT_MODEL_SOURCE = os.getenv("NOVA_MODEL_SOURCE", "").strip()
DEFAULT_CIVITAI_BASE_MODEL = os.getenv("NOVA_CIVITAI_BASE_MODEL", "Illustrious").strip()


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _load_flash_volume_types():
    try:
        from runpod_flash import DataCenter, NetworkVolume

        return DataCenter, NetworkVolume
    except Exception:  # pragma: no cover - compatibility with older or missing runpod-flash installs
        try:
            from runpod_flash.core.resources import DataCenter, NetworkVolume

            return DataCenter, NetworkVolume
        except Exception:  # pragma: no cover - optional at local app runtime
            return None, None


def shared_volume():
    data_center_type, network_volume_type = _load_flash_volume_types()
    if network_volume_type is None or data_center_type is None:  # pragma: no cover - guarded for local app installs
        raise RuntimeError("runpod-flash is required to build or deploy RunPod workers.")
    data_center = getattr(data_center_type, DEFAULT_DATA_CENTER, data_center_type.EU_RO_1)
    return network_volume_type(name=DEFAULT_VOLUME_NAME, size=DEFAULT_VOLUME_SIZE_GB, dataCenterId=data_center)


def volume_root() -> Path:
    return Path(os.getenv("RUNPOD_VOLUME_ROOT", "/runpod-volume"))


def endpoint_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "RUNPOD_VOLUME_ROOT": os.getenv("RUNPOD_VOLUME_ROOT", "/runpod-volume"),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    passthrough_keys = [
        "HF_TOKEN",
        "CIVITAI_API_TOKEN",
        "NOVA_MODEL_SOURCE",
        "NOVA_MODEL_SOURCE_MAP_JSON",
        "NOVA_MODEL_CACHE_DIR",
        "NOVA_SINGLE_FILE_PIPELINE_CLASS",
        "NOVA_TORCH_DTYPE",
        "NOVA_CIVITAI_BASE_MODEL",
        "SD_SCRIPTS_REPO",
        "SD_SCRIPTS_REF",
        "SD_SCRIPTS_AUTO_UPDATE",
        "SD_SCRIPTS_DIR",
        "LORA_CPU_THREADS",
        "LORA_LEARNING_RATE",
        "LORA_TEXT_ENCODER_LR",
        "LORA_UNET_LR",
        "LORA_SEED",
        "LORA_OUTPUT_NAME",
        "LORA_SAVE_PRECISION",
        "LORA_USE_SDPA",
        "LORA_USE_XFORMERS",
        "LORA_ENABLE_BUCKETS",
        "LORA_SAVE_STATE",
    ]
    for key in passthrough_keys:
        value = os.getenv(key, "").strip()
        if value:
            env[key] = value
    if extra:
        env.update(extra)
    return env


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def slugify(value: str, default: str = "artifact") -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    slug = "-".join(part for part in raw.split("-") if part)
    return slug or default


def hash_text(value: str, length: int = 10) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def resolve_model_source(base_model_ref: str) -> str:
    reference = base_model_ref.strip()
    if not reference:
        raise ValueError("`base_model_ref` is required.")

    source_map: dict[str, str] = {}
    raw_map = os.getenv("NOVA_MODEL_SOURCE_MAP_JSON", "").strip()
    if raw_map:
        source_map.update(json.loads(raw_map))
    if DEFAULT_MODEL_SOURCE:
        source_map.setdefault(DEFAULT_MODEL_ALIAS, DEFAULT_MODEL_SOURCE)

    return source_map.get(reference, reference)


def ensure_volume_path(path_str: str, *, create: bool = False, file_ok: bool = False) -> Path:
    value = path_str.strip()
    if not value:
        raise ValueError("Expected an absolute path on the shared RunPod volume.")
    path = Path(value)
    root = volume_root()
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    root = root.resolve(strict=False)
    if path != root and root not in path.parents:
        raise ValueError(f"Path must stay under {root}: {path}")
    if create:
        target = path if file_ok else path
        ensure_dir(target.parent if file_ok else target)
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def flatten_caption_tags(tags: list[str] | str) -> str:
    if isinstance(tags, str):
        raw_items = tags.split(",")
    else:
        raw_items = tags
    cleaned = [str(item).strip() for item in raw_items if str(item).strip()]
    return ", ".join(dict.fromkeys(cleaned))


def encode_file_base64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def link_or_copy(source: str | Path, target: str | Path) -> Path:
    src = Path(source)
    dst = Path(target)
    ensure_dir(dst.parent)
    if dst.exists():
        return dst
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def looks_like_url(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith("https://") or lowered.startswith("http://")


def _request_headers(url: str) -> dict[str, str]:
    headers = {
        "User-Agent": os.getenv("MODEL_DOWNLOAD_USER_AGENT", "Mozilla/5.0 AnimeLoRALab/1.0"),
        "Accept": "application/json, */*",
    }
    if "civitai.com" in urllib.parse.urlparse(url).netloc:
        token = os.getenv("CIVITAI_API_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=_request_headers(url))
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _filename_from_content_disposition(value: str) -> str:
    if not value:
        return ""
    star_match = re.search(r"filename\*\s*=\s*[^']*''([^;]+)", value, flags=re.IGNORECASE)
    if star_match:
        return urllib.parse.unquote(star_match.group(1)).strip().strip('"')
    plain_match = re.search(r'filename\s*=\s*"([^"]+)"', value, flags=re.IGNORECASE)
    if plain_match:
        return plain_match.group(1).strip()
    bare_match = re.search(r"filename\s*=\s*([^;]+)", value, flags=re.IGNORECASE)
    if bare_match:
        return bare_match.group(1).strip().strip('"')
    return ""


def _infer_download_filename(url: str, *, content_disposition: str = "") -> str:
    filename = _filename_from_content_disposition(content_disposition)
    if filename:
        return filename

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    for key in ("response-content-disposition", "b2ContentDisposition"):
        for value in query.get(key, []):
            filename = _filename_from_content_disposition(value)
            if filename:
                return filename

    return Path(parsed.path).name or "model.bin"


def resolve_civitai_download(model_url: str) -> tuple[str, str | None]:
    match = re.search(r"civitai\.com/models/(\d+)", model_url)
    if not match:
        return model_url, None

    model_id = match.group(1)
    payload = fetch_json(f"https://civitai.com/api/v1/models/{model_id}")
    versions = payload.get("modelVersions") or []
    filtered_versions = [item for item in versions if item.get("baseModel") == DEFAULT_CIVITAI_BASE_MODEL]
    chosen_versions = filtered_versions or versions
    if not chosen_versions:
        raise ValueError(f"No model versions found for Civitai model {model_id}.")
    file_entry = (chosen_versions[0].get("files") or [{}])[0]
    download_url = str(file_entry.get("downloadUrl") or "").strip()
    if not download_url:
        raise ValueError(f"Civitai model {model_id} does not expose a downloadable file.")
    filename = str(file_entry.get("name") or "").strip() or None
    return download_url, filename


def resolve_civitai_download_url(model_url: str) -> str:
    download_url, _ = resolve_civitai_download(model_url)
    return download_url


def _repair_cached_download(target_dir: Path, expected_filename: str) -> Path | None:
    expected = target_dir / expected_filename
    if expected.exists():
        return expected

    legacy_files = [item for item in target_dir.iterdir() if item.is_file()]
    if len(legacy_files) != 1:
        return None

    legacy = legacy_files[0]
    if legacy.name == expected_filename:
        return legacy
    if legacy.suffix != ".bin" or expected.suffix == ".bin":
        return None

    legacy.replace(expected)
    return expected


def download_to_cache(url: str, cache_root: str | Path) -> Path:
    resolved_url, preferred_filename = resolve_civitai_download(url)
    target_dir = ensure_dir(Path(cache_root) / hash_text(resolved_url, 16))
    if preferred_filename:
        repaired = _repair_cached_download(target_dir, preferred_filename)
        if repaired is not None:
            return repaired

    request = urllib.request.Request(resolved_url, headers=_request_headers(resolved_url))
    with urllib.request.urlopen(request) as response:
        filename = preferred_filename or _infer_download_filename(
            response.geturl(),
            content_disposition=response.headers.get("Content-Disposition", ""),
        )
        if filename.isdigit():
            filename = f"{filename}.bin"
        repaired = _repair_cached_download(target_dir, filename)
        if repaired is not None:
            return repaired
        target = target_dir / filename
        if target.exists():
            return target
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return target


def snapshot_hf_repo(repo_id: str, cache_root: str | Path) -> Path:
    from huggingface_hub import snapshot_download

    target = ensure_dir(Path(cache_root) / slugify(repo_id, "hf-model"))
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        token=os.getenv("HF_TOKEN") or None,
    )
    return Path(snapshot_path)


def prepare_local_model_source(base_model_ref: str, cache_root: str | Path) -> Path:
    resolved = resolve_model_source(base_model_ref)
    candidate = Path(resolved)
    if candidate.exists():
        return candidate
    if looks_like_url(resolved):
        return download_to_cache(resolved, cache_root)
    if "/" in resolved and " " not in resolved and not resolved.endswith(".safetensors"):
        return snapshot_hf_repo(resolved, cache_root)
    raise ValueError(
        f"Could not resolve base model source '{resolved}'. "
        "Set NOVA_MODEL_SOURCE or NOVA_MODEL_SOURCE_MAP_JSON to a local path, URL, or Hugging Face repo ID."
    )


def run_logged_command(
    command: list[str],
    *,
    log_path: str | Path,
    cwd: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    ensure_dir(Path(log_path).parent)
    with Path(log_path).open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def newest_matching_file(root: str | Path, pattern: str) -> Path | None:
    matches = sorted(Path(root).glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0] if matches else None
