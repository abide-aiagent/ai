from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class Settings(BaseSettings):
    """AI Server configuration"""

    # Database
    database_url: str = os.getenv("DATABASE_URL")

    # Redis
    redis_url: str = os.getenv("REDIS_URL")

    # LLM
    openai_api_key: str = ""
    gemini_api_key: str = os.getenv("GEMINI_API_KEY")

    # Models
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/text-embedding-004"
    embedding_dimensions: int = 768

    # Server
    ai_server_port: int = 8000
    core_server_url: str = "http://localhost:8080"
    log_level: str = "info"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
