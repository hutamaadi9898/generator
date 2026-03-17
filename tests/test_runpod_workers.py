from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runpod_workers.nova_generate import core as nova_generate_core
from runpod_workers.common import ensure_volume_path, flatten_caption_tags, resolve_model_source


def test_flatten_caption_tags_dedupes_and_joins() -> None:
    assert flatten_caption_tags(["aoi", "portrait", "aoi", " soft smile "]) == "aoi, portrait, soft smile"


def test_resolve_model_source_uses_alias_map(monkeypatch) -> None:
    monkeypatch.setenv("NOVA_MODEL_SOURCE_MAP_JSON", '{"nova-anime-xl-illustrious":"hf-org/nova-anime-xl"}')
    assert resolve_model_source("nova-anime-xl-illustrious") == "hf-org/nova-anime-xl"


def test_ensure_volume_path_rejects_escape(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNPOD_VOLUME_ROOT", str(tmp_path))
    allowed = ensure_volume_path(str(tmp_path / "characters" / "1"), create=True)
    assert allowed.exists()


def test_nova_generate_load_pipeline_resolves_remote_model_to_local_file(monkeypatch, tmp_path: Path) -> None:
    model_file = tmp_path / "nova-model.safetensors"
    model_file.write_bytes(b"stub")

    class FakeScheduler:
        config = {"name": "stub"}

    class FakePipeline:
        def __init__(self):
            self.scheduler = FakeScheduler()

        def enable_xformers_memory_efficient_attention(self):
            return None

        def enable_vae_tiling(self):
            return None

        def enable_attention_slicing(self):
            return None

        def to(self, device: str):
            self.device = device
            return self

    class FakeAutoPipeline:
        called = None

        @classmethod
        def from_single_file(cls, path: str, **kwargs):
            cls.called = ("single", path, kwargs)
            return FakePipeline()

        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            cls.called = ("pretrained", path, kwargs)
            return FakePipeline()

    class FakeEulerScheduler:
        @staticmethod
        def from_config(config):
            return FakeScheduler()

    class FakeTorch:
        float16 = "float16"

    monkeypatch.setenv("NOVA_MODEL_CACHE_DIR", str(tmp_path / "model-cache"))
    monkeypatch.setattr(nova_generate_core, "prepare_local_model_source", lambda base_model_ref, cache_root: model_file)
    monkeypatch.setattr(nova_generate_core, "resolve_model_source", lambda base_model_ref: "https://civitai.com/models/376130/nova-anime-xl")
    nova_generate_core._PIPELINES.clear()

    pipeline = nova_generate_core._load_pipeline(
        FakeAutoPipeline,
        FakeEulerScheduler,
        FakeTorch,
        "nova-anime-xl-illustrious",
    )

    assert isinstance(pipeline, FakePipeline)
    assert FakeAutoPipeline.called is not None
    mode, path, kwargs = FakeAutoPipeline.called
    assert mode == "single"
    assert path == str(model_file)
    assert kwargs["use_safetensors"] is True
