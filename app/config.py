import os
from dataclasses import dataclass


@dataclass
class Settings:
    hf_token: str = os.getenv("HF_TOKEN", "")
    code_model_id: str = os.getenv(
        "CODE_MODEL_ID", "bigcode/starcoder2-3b"
    )  # lighter for free tier
    diffusion_model_id: str = os.getenv(
        "DIFFUSION_MODEL_ID", "stabilityai/stable-diffusion-2-1"
    )
    database_path: str = os.getenv("DATABASE_PATH", os.path.join("data", "app.db"))
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))


settings = Settings()




