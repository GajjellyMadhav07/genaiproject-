import base64
from typing import Dict, Any
import httpx

from app.config import settings


HF_API_URL = "https://api-inference.huggingface.co/models"


class HFInferenceClient:
    def __init__(self, token: str | None = None) -> None:
        self.token = token or settings.hf_token
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def text_generation(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{HF_API_URL}/{model_id}"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def text_to_image(self, model_id: str, prompt: str) -> bytes:
        url = f"{HF_API_URL}/{model_id}"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, headers=self.headers, data=prompt)
            resp.raise_for_status()
            return resp.content

    @staticmethod
    def image_bytes_to_base64(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")




