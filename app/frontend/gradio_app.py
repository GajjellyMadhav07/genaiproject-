import base64
import json
import asyncio
from typing import Any, Dict, Tuple

import gradio as gr

from app.services.codegen import generate_code
from app.services.diagram import generate_diagram
from app.services.analysis import analyze_code
from app.metrics.metrics import timer
from app.services.hf_clients import HFInferenceClient
from app.storage.db import save_message, fetch_history


async def orchestrate(prompt: str, session_id: str) -> Tuple[str, str, Dict[str, Any]]:
    with timer() as (_, elapsed_total):
        # code gen timing
        with timer() as (_, elapsed_code):
            code = await generate_code(prompt)
        # diagram timing
        with timer() as (_, elapsed_img):
            diagram_bytes = await generate_diagram(prompt, code)
        diagram_b64 = HFInferenceClient.image_bytes_to_base64(diagram_bytes)
        # analysis timing
        with timer() as (_, elapsed_ana):
            analysis = analyze_code(code or "")
        # attach basic latency metrics (ms)
        analysis["latency_ms"] = {
            "total": elapsed_total(),
            "codegen": elapsed_code(),
            "diagram": elapsed_img(),
            "analysis": elapsed_ana(),
        }
    save_message(
        session_id=session_id,
        user_prompt=prompt,
        generated_code=code,
        diagram_base64=diagram_b64,
        analysis=analysis,
    )
    return code, diagram_b64, analysis


def _orchestrate_sync(prompt: str, session_id: str):
    return asyncio.run(orchestrate(prompt, session_id))


def _history_sync(session_id: str):
    return fetch_history(session_id)


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Multi-Model Code Assistant") as demo:
        session_state = gr.State("default")

        gr.Markdown("""# Code Generation Assistant
Enter your request to generate code, architecture diagram, and analysis.
""")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Generate Fibonacci sequence in Java")
        with gr.Row():
            session = gr.Textbox(label="Session ID", value="default")
        with gr.Row():
            submit = gr.Button("Generate")

        with gr.Tab("Results"):
            code_out = gr.Code(language="python", label="Generated Code")
            image_out = gr.Image(label="Architecture Diagram")
            analysis_out = gr.JSON(label="Code Analysis")

        with gr.Tab("History"):
            history_session = gr.Textbox(label="Session ID", value="default")
            reload = gr.Button("Reload History")
            history_json = gr.JSON(label="History")

        def _on_submit(p: str, s: str):
            code, diagram_b64, analysis = _orchestrate_sync(p, s)
            img_bytes = base64.b64decode(diagram_b64)
            return code, img_bytes, analysis

        submit.click(_on_submit, inputs=[prompt, session], outputs=[code_out, image_out, analysis_out])

        def _on_reload(s: str):
            return _history_sync(s)

        reload.click(_on_reload, inputs=[history_session], outputs=[history_json])

    return demo




