from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runpod_workers import common as common_module
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


def test_nova_generate_load_pipeline_falls_back_when_auto_pipeline_lacks_single_file(
    monkeypatch, tmp_path: Path
) -> None:
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
        def from_pretrained(cls, path: str, **kwargs):
            cls.called = ("pretrained", path, kwargs)
            return FakePipeline()

    class FakeXLPipeline:
        called = None

        @classmethod
        def from_single_file(cls, path: str, **kwargs):
            cls.called = ("single", path, kwargs)
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
    monkeypatch.setattr(
        nova_generate_core,
        "_resolve_single_file_pipeline_classes",
        lambda base_model_ref, resolved_model_source: [FakeXLPipeline],
    )
    nova_generate_core._PIPELINES.clear()

    pipeline = nova_generate_core._load_pipeline(
        FakeAutoPipeline,
        FakeEulerScheduler,
        FakeTorch,
        "nova-anime-xl-illustrious",
    )

    assert isinstance(pipeline, FakePipeline)
    assert FakeAutoPipeline.called is None
    assert FakeXLPipeline.called is not None
    mode, path, kwargs = FakeXLPipeline.called
    assert mode == "single"
    assert path == str(model_file)
    assert kwargs["use_safetensors"] is True


def test_resolve_single_file_pipeline_classes_prefers_sdxl_for_nova(monkeypatch) -> None:
    class FakeSDPipeline:
        @classmethod
        def from_single_file(cls, path: str, **kwargs):
            raise NotImplementedError

    class FakeSDXLPipeline:
        @classmethod
        def from_single_file(cls, path: str, **kwargs):
            raise NotImplementedError

    class FakeDiffusers:
        StableDiffusionPipeline = FakeSDPipeline
        StableDiffusionXLPipeline = FakeSDXLPipeline

    monkeypatch.setitem(sys.modules, "diffusers", FakeDiffusers)

    resolved = nova_generate_core._resolve_single_file_pipeline_classes(
        "nova-anime-xl-illustrious",
        "https://civitai.com/models/376130/nova-anime-xl",
    )

    assert resolved == [FakeSDXLPipeline, FakeSDPipeline]


def test_resolve_civitai_download_includes_named_file(monkeypatch) -> None:
    monkeypatch.setattr(
        common_module,
        "fetch_json",
        lambda url: {
            "modelVersions": [
                {
                    "baseModel": "Illustrious",
                    "files": [
                        {
                            "name": "novaAnimeXL_ilV170.safetensors",
                            "downloadUrl": "https://civitai.com/api/download/models/2741698",
                        }
                    ],
                }
            ]
        },
    )

    resolved_url, filename = common_module.resolve_civitai_download("https://civitai.com/models/376130/nova-anime-xl")

    assert resolved_url == "https://civitai.com/api/download/models/2741698"
    assert filename == "novaAnimeXL_ilV170.safetensors"


def test_download_to_cache_repairs_legacy_numeric_bin(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    resolved_url = "https://civitai.com/api/download/models/2741698"
    target_dir = cache_root / common_module.hash_text(resolved_url, 16)
    target_dir.mkdir(parents=True)
    legacy_file = target_dir / "2741698.bin"
    legacy_file.write_bytes(b"stub")

    monkeypatch.setattr(
        common_module,
        "resolve_civitai_download",
        lambda url: (resolved_url, "novaAnimeXL_ilV170.safetensors"),
    )

    resolved = common_module.download_to_cache("https://civitai.com/models/376130/nova-anime-xl", cache_root)

    assert resolved == target_dir / "novaAnimeXL_ilV170.safetensors"
    assert resolved.read_bytes() == b"stub"
    assert not legacy_file.exists()


def test_infer_download_filename_uses_content_disposition() -> None:
    filename = common_module._infer_download_filename(
        "https://b2.civitai.com/file/civitai-modelfiles/model/2515131/novaanimeilv17.Gii7.safetensors"
        "?b2ContentDisposition=attachment%3B+filename%3D%22novaAnimeXL_ilV170.safetensors%22",
        content_disposition='attachment; filename="novaAnimeXL_ilV170.safetensors"',
    )

    assert filename == "novaAnimeXL_ilV170.safetensors"
