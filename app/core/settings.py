from __future__ import annotations

from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Config central da aplicação.

    - Em produção, você normalmente injeta env vars via runtime (Docker/K8s/Render/etc.).
    - Em dev, pode usar .env (por padrão).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ambiente
    env: str = Field(default="development", alias="ENV")

    # API
    title: str = Field(default="AI REVENUE MACHINE", alias="APP_TITLE")
    version: str = Field(default="0.1.0", alias="APP_VERSION")

    # CORS
    allow_origins_raw: str = Field(default="*", alias="ALLOW_ORIGINS")
    allow_credentials: bool = Field(default=False, alias="ALLOW_CREDENTIALS")

    # Database
    database_url: str = Field(..., alias="DATABASE_URL")
    db_pool_min_size: int = Field(default=1, alias="DB_POOL_MIN_SIZE")
    db_pool_max_size: int = Field(default=10, alias="DB_POOL_MAX_SIZE")

    # Agent / LLM
    debug_agent_logs: bool = Field(default=False, alias="DEBUG_AGENT_LOGS")

    default_model_name: str = Field(default="google/gemini-2.5-flash", alias="DEFAULT_MODEL_NAME")
    default_use_tavily: bool = Field(default=False, alias="DEFAULT_USE_TAVILY")

    studio_model_name: Optional[str] = Field(default=None, alias="STUDIO_MODEL_NAME")
    studio_use_tavily: Optional[bool] = Field(default=None, alias="STUDIO_USE_TAVILY")

    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")

    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")


    @property
    def allow_origins(self) -> List[str]:
        raw = (self.allow_origins_raw or "").strip()
        if not raw:
            return []
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]
    
    @property
    def effective_model_name(self) -> str:
        return self.studio_model_name or self.default_model_name

    @property
    def effective_use_tavily(self) -> bool:
        return self.default_use_tavily if self.studio_use_tavily is None else self.studio_use_tavily

    @field_validator("allow_credentials")
    @classmethod
    def _validate_cors_credentials(cls, v: bool, info):
        # Regra importante: credentials=true não pode combinar com allow_origins="*"
        # (navegadores bloqueiam e é inseguro).
        raw = (info.data.get("allow_origins_raw") or "").strip()
        if v and raw == "*":
            raise ValueError('ALLOW_CREDENTIALS=true não pode ser usado com ALLOW_ORIGINS="*".')
        return v


def get_settings() -> Settings:
    return Settings()
