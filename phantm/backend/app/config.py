"""
App configuration loaded from environment variables.

Uses pydantic-settings so every value is validated at startup.
All env vars are documented in .env.example.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Validated app settings from environment variables."""

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""

    # JWT (Supabase uses its own JWT secret internally)
    jwt_secret: str = ""

    # Cloudflare R2
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = "liveself-assets"

    # Engine communication
    engine_url: str = "http://localhost:8001"
    engine_api_key: str = ""

    # App
    app_env: str = "development"
    frontend_url: str = "http://localhost:3000"
    backend_url: str = "http://localhost:8000"

    model_config = {"env_file": ".env", "extra": "ignore"}


def get_settings() -> Settings:
    """Returns validated settings. Called once at startup."""
    return Settings()
