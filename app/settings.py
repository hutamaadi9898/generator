from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Anime LoRA Lab"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    app_base_url: str = "http://127.0.0.1:8000"
    data_dir: Path = Field(default=Path("data"))
    database_url: str = "sqlite:///./data/app.db"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    gemini_api_base: str = "https://generativelanguage.googleapis.com/v1beta"

    runpod_api_key: str = ""
    runpod_api_base: str = "https://api.runpod.ai/v2"
    runpod_generate_endpoint_id: str = ""
    runpod_train_endpoint_id: str = ""
    runpod_submission_mode: str = "plain"
    runpod_generate_function_name: str = "nova_generate"
    runpod_train_function_name: str = "lora_train"
    runpod_volume_root: str = "/runpod-volume"

    base_model_ref: str = "nova-anime-xl-illustrious"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
