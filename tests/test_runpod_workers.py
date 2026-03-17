from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
