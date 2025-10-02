from app.config import settings
from app.services.hf_clients import HFInferenceClient


def build_diagram_prompt(user_prompt: str, generated_code: str | None) -> str:
    desc = user_prompt
    if generated_code:
        desc += "\n\nSummarize code architecture: identify components, data flow, and interfaces."
    return (
        "Create a clear systems architecture diagram. Show components and arrows for data flow.\n"
        f"Context: {desc}"
    )


async def generate_diagram(user_prompt: str, generated_code: str | None) -> bytes:
    client = HFInferenceClient()
    prompt = build_diagram_prompt(user_prompt, generated_code)
    image_bytes = await client.text_to_image(settings.diffusion_model_id, prompt)
    return image_bytes




