from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "dev"
    app_secret: str = "replace-me"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/termalign"
    redis_url: str = "redis://localhost:6379/0"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    cors_origins: str = "http://localhost:5173,http://localhost:5174"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
