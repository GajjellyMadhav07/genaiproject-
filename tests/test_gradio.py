import importlib
import os
import tempfile
import base64


def test_gradio_orchestration(monkeypatch):
    # Use a temp DB path before importing gradio app
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("DATABASE_PATH", path)

    # Mock services
    async def fake_generate_code(prompt: str) -> str:
        return "print('ok')\n"

    async def fake_generate_diagram(prompt: str, code: str | None) -> bytes:
        return base64.b64decode(base64.b64encode(b"img"))

    from app.services import codegen as cg
    from app.services import diagram as dg

    monkeypatch.setattr(cg, "generate_code", fake_generate_code)
    monkeypatch.setattr(dg, "generate_diagram", fake_generate_diagram)

    from app.frontend import gradio_app as ga

    importlib.reload(ga)
    code, b64, analysis = ga._orchestrate_sync("Say hi", "s1")
    assert code.startswith("print")
    assert isinstance(b64, str)
    assert "token_count" in analysis


