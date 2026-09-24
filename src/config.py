from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Database Settings (TimescaleDB) ---
    DB_USER: str = Field(default="admin")
    DB_PASSWORD: str = Field(default="sentinel_secure_pass_2026")
    DB_NAME: str = Field(default="airsentinel_db")
    DB_HOST: str = Field(default="localhost")
    DB_PORT: int = Field(default=5432)

    # --- Cache Settings (Redis) ---
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)

    # --- Storage Settings (MinIO) ---
    MINIO_USER: str = Field(default="storage_admin")
    MINIO_PASSWORD: str = Field(default="minio_secret_vault_key")
    MINIO_ENDPOINT: str = Field(default="http://localhost:9000")

    # --- MLOps Settings (MLflow) ---
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000")

    # --- LLM Settings (Groq) ---
    # Single source of truth for the Groq model name. Groq deprecates models
    # with real advance notice but no fixed cadence (llama-3.1-8b-instant and
    # llama-3.3-70b-versatile were both deprecated 2026-06-17) — when the next
    # one lands, change it HERE only, not in react_agent.py or
    # report_generator.py separately.
    #
    # FIX: this had been reverted back to llama-3.3-70b-versatile (the exact
    # deprecated model documented in this comment) — likely a merge or a
    # manual edit that undid the earlier fix. Restored to the confirmed-
    # working replacement. If you see 404 model_not_found again, that's the
    # first thing to check — someone/something is overwriting this line.
    GROQ_MODEL: str = Field(default="openai/gpt-oss-20b")
    GROQ_API_KEY: str | None = Field(default=None)

    # --- LLM Fallback Settings (Google Gemini) ---
    GEMINI_API_KEY: str | None = Field(default=None)
    GOOGLE_API_KEY: str | None = Field(default=None)
    GEMINI_MODEL: str = Field(default="gemini-3.6-flash")

    # --- Auth Settings (JWT) ---
    JWT_SECRET_KEY: str = Field(default="CHANGE_ME_IN_ENV")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRE_MINUTES: int = Field(default=60)
    ADMIN_USERNAME: str = Field(default="admin")
    ADMIN_PASSWORD_HASH: str = Field(default="")

    # --- Compiled Connection Helpers ---
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()