"""Application configuration, loaded from environment variables (prefix ``PA_``)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PA_", env_file=".env", extra="ignore")

    environment: str = "local"
    rationale_mode: str = "template"  # template | llm
    llm_providers: str = "anthropic,openai,gemini,deepseek,ollama"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
