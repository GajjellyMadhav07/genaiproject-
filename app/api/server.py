import asyncio
import json
from typing import Any, Dict

from flask import Flask, request, jsonify

from app.config import settings
from app.services.codegen import generate_code
from app.services.diagram import generate_diagram
from app.services.analysis import analyze_code
from app.storage.db import save_message, fetch_history
from app.services.hf_clients import HFInferenceClient


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/api/health", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.route("/api/generate", methods=["POST"])
    def generate() -> Any:
        data: Dict[str, Any] = request.get_json(force=True)
        user_prompt: str = data.get("prompt", "")
        session_id: str = data.get("session_id", "default")

        async def _run():
            code_task = asyncio.create_task(generate_code(user_prompt))
            # diagram can use code later; await code first for better context
            code = await code_task
            diagram_bytes = await generate_diagram(user_prompt, code)
            diagram_b64 = HFInferenceClient.image_bytes_to_base64(diagram_bytes)
            analysis = analyze_code(code or "")
            return code, diagram_b64, analysis

        code, diagram_b64, analysis = asyncio.run(_run())

        save_message(
            session_id=session_id,
            user_prompt=user_prompt,
            generated_code=code,
            diagram_base64=diagram_b64,
            analysis=analysis,
        )

        return jsonify(
            {
                "generated_code": code,
                "diagram_base64": diagram_b64,
                "analysis": analysis,
            }
        )

    @app.route("/api/history", methods=["GET"])
    def history() -> Any:
        session_id: str = request.args.get("session_id", "default")
        items = fetch_history(session_id=session_id)
        return jsonify(items)

    return app


def main() -> None:
    app = create_app()
    app.run(host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()




