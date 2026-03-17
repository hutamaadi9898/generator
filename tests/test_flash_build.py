from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


runpod_flash_build = pytest.importorskip("runpod_flash.cli.commands.build")


def test_flash_extracts_literal_endpoint_dependencies() -> None:
    build_dir = Path(__file__).resolve().parents[1] / "runpod_workers"

    requirements = runpod_flash_build.extract_remote_dependencies(build_dir)

    assert "torch>=2.5.1" in requirements
    assert "diffusers>=0.35.0" in requirements
    assert "transformers>=4.48.0" in requirements
    assert "accelerate>=1.2.1" in requirements
    assert "huggingface_hub>=0.29.0" in requirements
    assert "safetensors>=0.4.5" in requirements
    assert "sentencepiece>=0.2.0" in requirements
    assert "Pillow>=11.0.0" in requirements
