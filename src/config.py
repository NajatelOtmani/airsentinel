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

    # --- Compiled Connection Helpers ---
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
