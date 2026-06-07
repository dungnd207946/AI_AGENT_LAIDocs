"""LAIDocs configuration management using pydantic-settings with JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

LAIDOCS_HOME = Path.home() / ".laidocs"
CONFIG_PATH = LAIDOCS_HOME / "config.json"


class LLMConfig(BaseModel):
    # provider: "" resolves to "openai". Supported: "openai" (OpenAI-compatible,
    # incl. local/Ollama/LM Studio), "gemini" (Google), "anthropic".
    provider: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class Settings(BaseSettings):
    """LAIDocs application settings persisted to ~/.laidocs/config.json and loaded from .env."""

    llm: LLMConfig = LLMConfig()
    port: int = 8008
    telemetry_url: str = "http://localhost:8001/api/v1/track"
    telemetry_enabled: bool = True

    default_llm_provider: str = ""
    default_llm_base_url: str = ""
    default_llm_api_key: str = ""
    default_llm_model: str = ""

    @property
    def active_llm(self) -> LLMConfig:
        return LLMConfig(
            provider=self.llm.provider or self.default_llm_provider,
            base_url=self.llm.base_url or self.default_llm_base_url,
            api_key=self.llm.api_key or self.default_llm_api_key,
            model=self.llm.model or self.default_llm_model,
        )

    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def save_to_file(self, path: Path | None = None) -> None:
        target = path or CONFIG_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_from_file(cls, path: Path | None = None) -> Settings:
        target = path or CONFIG_PATH
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            return cls()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            # Filter out removed keys for backward compat
            valid_keys = cls.model_fields.keys()
            filtered = {k: v for k, v in raw.items() if k in valid_keys}
            return cls.model_validate(filtered)
        except (json.JSONDecodeError, Exception):
            return cls()


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load_from_file()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings.load_from_file()
    return _settings
