from typing import Dict, Any

from app.config import settings
from app.services.hf_clients import HFInferenceClient


SYSTEM_PROMPT = (
    "You are a code generation assistant. Generate clean, runnable code with brief comments."
)


def build_codegen_prompt(user_prompt: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser request:\n{user_prompt}\n\nProvide only code when appropriate."


async def generate_code(user_prompt: str) -> str:
    client = HFInferenceClient()
    payload: Dict[str, Any] = {
        "inputs": build_codegen_prompt(user_prompt),
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.9,
            "return_full_text": False,
        },
    }
    result = await client.text_generation(settings.code_model_id, payload)
    # HF text generation response may vary; handle common structures
    if isinstance(result, list) and result and "generated_text" in result[0]:
        return result[0]["generated_text"]
    if isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"]
    # Fallback: string or raw
    return str(result)




